from functools import reduce
from math import prod
from dataclasses import dataclass
import envpool
import gymnasium as gym
import gym as ogym
from gym import spaces as gspaces
from gymnasium import spaces
import jax
import numpy as np
from jac.asynclog import create_logger

from jac.gym_dmc import bound


class ParaGym:
    observation_space: gym.Space
    action_space: gym.Space

    def reset(self, index=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError


def flattener(x, keepdim):
    if isinstance(x, (spaces.Box, ogym.spaces.Box)):
        return IdenFlat(x)
    elif isinstance(x, (spaces.Dict, ogym.spaces.Dict)):
        return DictFlat(x, keepdim)
    print(type(x))
    raise NotImplementedError(type(x))


class IdenFlat:
    def __init__(self, x):
        self.space = x

    def flat(self, x):
        return x

    def unflat(self, x):
        return x


class DictFlat:
    def __init__(self, x: spaces.Dict, keepdims=0):
        self.dtype = reduce(np.promote_types, (i.dtype for i in x.values()))
        self.keys = list(x.keys())
        lows = []
        highs = []
        self.shapes = {}
        shape = 0

        for k in self.keys:
            i = x[k]
            if not isinstance(i, (spaces.Box, gspaces.Box)):
                raise NotImplementedError
            p = prod(i.shape)
            shape += p
            self.shapes[k] = i.shape
            if isinstance(i.low, np.ndarray):
                lows.append(i.low.astype(self.dtype).flatten())
            else:
                lows.append(np.full((p,), i.low, self.dtype))

            if isinstance(i.high, np.ndarray):
                highs.append(i.high.astype(self.dtype).flatten())
            else:
                highs.append(np.full((p,), i.high, self.dtype))
        low = bound(np.concatenate(lows))
        high = bound(np.concatenate(highs))
        self.space = spaces.Box(low, high, dtype=self.dtype, shape=(shape,))
        self.keepdims = keepdims

    def flat(self, x):
        return np.concatenate(
            [x[i].reshape((*x[i].shape[: self.keepdims], -1)) for i in self.keys],
            axis=-1,
        )

    def unflat(self, x):
        res = {}
        for k in self.keys:
            v = x[k]
            res[k] = v.reshape(*v.shape[:-1], *self.shapes[k])
        return res


class Scaler:
    def __init__(self, x: spaces.Box, scale):
        self.x = x
        self.scale = scale
        self.ospace = x
        self.space = spaces.Box(
            x.low * scale, x.high * scale, dtype=x.dtype, shape=x.shape
        )

    def transform(self, x):
        return x * self.scale

    def inverse_transform(self, x):
        return np.clip(x / self.scale, self.ospace.low, self.ospace.high)


@dataclass
class EnvConf:
    id: str
    num_envs: int
    num_envs_test: int
    scale: float = 1.0

    def __call__(self) -> tuple[ParaGym, ParaGym]:
        raise NotImplementedError


@dataclass
class EnvPoolConf(EnvConf):
    def __call__(self, seed1, seed2):
        if self.id not in envpool.list_all_envs():
            print("falling back to gym")
            return MultiGymConf.__call__(self, seed1, seed2)
        return WrapperEnvpool(
            num_envs=self.num_envs, scale=self.scale, id=self.id, seed=seed1
        ), WrapperEnvpool(
            self.num_envs_test,
            scale=self.scale,
            id=self.id,
            seed=max(seed2, seed1 + self.num_envs + 1),
        )


@dataclass
class MultiGymConf(EnvConf):
    def __call__(self, seed1, seed2):
        return WrapperMultiGym(
            num_envs=self.num_envs, id=self.id, scale=self.scale, seed=seed1
        ), WrapperMultiGym(
            num_envs=self.num_envs_test, scale=self.scale, id=self.id, seed=seed2
        )


class WrapperEnvpool(ParaGym):
    def __init__(self, num_envs, scale, id, **kwargs) -> None:
        self.env = envpool.make_gym(
            num_envs=num_envs, task_id=id, num_threads=2, **kwargs
        )
        self.fo = flattener(self.env.observation_space, 1)
        self.fa = flattener(self.env.action_space, 1)
        self.scaler = Scaler(self.fa.space, scale)

        self.observation_space = self.fo.space
        self.action_space = self.scaler.space

    def reset(self, index=None):
        obs, info = self.env.reset(index)
        return self.fo.flat(obs), info

    def step(self, actions):
        if isinstance(actions, jax.Array):
            actions = np.asarray(actions)
        actions = self.fa.unflat(self.scaler.inverse_transform(actions))
        obs, r, term, trunc, info = self.env.step(actions)
        obs = self.fo.flat(obs)
        return obs, r, term, trunc, info


class WrapperMultiGym(ParaGym):
    def __init__(self, num_envs, scale, seed=None, **kwargs):
        self.envs = [gym.make(**kwargs) for _ in range(num_envs)]
        self.fo = flattener(self.envs[0].observation_space, 0)
        self.fa = flattener(self.envs[0].action_space, 0)
        self.scaler = Scaler(self.fa.space, scale)
        self.observation_space = self.fo.space
        self.action_space = self.scaler.space
        if seed is not None:
            self.seed(self._seed(seed))

    def _seed(self, seed):
        rng = np.random.default_rng(seed)
        seed = rng.integers(100, 100000, size=len(self.envs))
        return [int(i) for i in seed]

    def seed(self, seed):
        assert len(seed) == len(self.envs)
        for i, j in zip(self.envs, seed):
            i.reset(seed=j)

    def reset(self, index=None):
        if index is None:
            index = range(len(self.envs))
        assert len(index) > 0
        obs = [self.fo.flat(self.envs[i].reset()[0]) for i in index]

        return np.stack(obs, axis=0), {}

    def step(self, actions):
        assert actions.shape[0] == len(self.envs)

        obs, r, term, trunc, _ = zip(
            *[
                e.step(self.fa.unflat(self.scaler.inverse_transform(actions)))
                for i, e in enumerate(self.envs)
            ]
        )
        obs = [self.fo.flat(i) for i in obs]
        obs = np.stack(obs, axis=0)
        return (
            obs,
            np.stack(r, axis=0),
            np.stack(term, axis=0),
            np.stack(trunc, axis=0),
            {},
        )
