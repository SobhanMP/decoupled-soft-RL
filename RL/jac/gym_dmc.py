from collections import OrderedDict
from dm_env import specs
from dm_control import suite
import gymnasium as gym
from gymnasium import spaces
import numpy as np


def bound(x):
    if all(i == x[0] for i in x):
        return x[0]
    else:
        return np.array(x)


def unbound(shape, dtype, x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return np.full(shape, x, dtype=dtype)


def space_to_spec(x):
    if isinstance(x, spaces.Box):
        minimum = unbound(x.shape, x.dtype, x.low)
        maximum = unbound(x.shape, x.dtype, x.high)
        if np.isneginf(minimum).all() and np.isinf(maximum).all():
            return specs.Array(
                dtype=x.dtype,
                shape=x.shape,
            )
        else:
            return specs.BoundedArray(
                minimum=minimum,
                maximum=maximum,
                dtype=x.dtype,
                shape=x.shape,
            )
    if isinstance(x, spaces.Dict):
        pass

    raise NotImplementedError(str(type(x)))


def spec_to_space(x):
    if isinstance(x, specs.BoundedArray):
        low = bound(x.minimum)
        high = bound(x.maximum)
        shape = x.shape
        return spaces.Box(low, high, dtype=x.dtype, shape=shape)

    if isinstance(x, specs.Array):
        low = -np.inf
        high = np.inf
        shape = x.shape
        return spaces.Box(low, high, dtype=x.dtype, shape=shape)
    if isinstance(x, OrderedDict):
        x = OrderedDict([(k, spec_to_space(v)) for k, v in x.items()])
        return spaces.Dict(x)
    raise NotImplementedError(str(type(x)))


class DMC(gym.Env):
    def __init__(self, domain_name, task_name, seed=None, max_steps=None):
        self.max_steps = max_steps
        self.domain_name = domain_name
        self.task_name = task_name
        self.load_env(seed)
        self.reset(seed)
        self.action_space = spec_to_space(self.env.action_spec())
        self.observation_space = spec_to_space(self.env.observation_spec())

    def load_env(self, seed):
        if hasattr(self, "env"):
            self.env.close()  # pylint: disable=access-member-before-definition
        self.env = suite.load(
            self.domain_name,
            self.task_name,
            task_kwargs={"random": seed},
        )

    def reset(self, *_, seed=None, options=None):
        if seed is not None:
            self.load_env(seed)
        self.steps = 0
        ts = self.env.reset()
        return ts.observation, dict()

    def step(self, action):
        self.steps += 1
        ts = self.env.step(action)
        obs = ts.observation
        last = ts.last()
        if not last and self.max_steps is not None and self.steps >= self.max_steps:
            trunc = True
        else:
            trunc = last and ts.discount == 1
        term = last and (not trunc)
        # print("term, trunc", term, trunc)
        assert not (term and trunc)
        if last:
            assert term or trunc
        return (obs, ts.reward, term, trunc, dict())

    def render(self):
        return None

    def close(self):
        self.env.close()


def conv_str(x):
    res = [x[0].upper()]
    i = 1
    while i < len(x):
        if x[i] == "_":
            if i + 1 < len(x):
                res.append(x[i + 1].upper())
                i += 2
        else:
            res.append(x[i])
            i += 1
    return "".join(res)


def name(d, t):
    return conv_str(d) + conv_str(t) + "-v1"


BENCHMARKING = []
ALL_TASKS = []


def load_benchmarks():
    for d, t in suite.ALL_TASKS:
        i = name(d, t)
        ALL_TASKS.append(i)
        gym.register(
            i,
            entry_point="jac.gym_dmc:DMC",
            kwargs={
                "domain_name": d,
                "task_name": t,
            },
        )

    for d, t in suite.BENCHMARKING:
        BENCHMARKING.append(name(d, t))


load_benchmarks()
