from contextlib import closing
import itertools
from math import ceil
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import omegaconf

from jac.algo import AlgoConf, Algorithm

from jac.consts import np
from jac.asynclog import LogConf
from jac.env_wrapper import EnvConf, ParaGym
from jac.replay import RBConf
from jac.utils import JRK, Erec, Seeder
from jac.asynclog import Logger
import jac.gym_dmc
from omegaconf import MISSING

__all__ = ["Conf", "train"]


@dataclass(eq=True, kw_only=True)
class TrConf:
    batch_size: int = MISSING
    steps_per_epoch: int = MISSING
    steps: int = MISSING
    replay_ratio: int = MISSING
    reset_every: Optional[int] = MISSING
    random_period: int = MISSING
    start_training: int = MISSING
    test_episodes: int = MISSING


@dataclass(eq=True, kw_only=True)
class Conf:
    algo: AlgoConf = MISSING
    log: LogConf = MISSING
    train: TrConf = MISSING
    replay: RBConf = MISSING
    env: EnvConf = MISSING
    seed: Optional[int] = MISSING


def test_agent(cfg: Conf, algo: Algorithm, env: ParaGym):
    res = defaultdict(list)
    test_start_time = time.time()
    obs, _ = env.reset()
    rec = Erec(obs.shape[0])
    n_done = 0
    while n_done < cfg.train.test_episodes:
        obs, reward, term, trunc, _ = env.step(algo.act(None, obs))

        done = term | trunc
        done_id = np.where(done)[0]
        if len(done_id) > 0:
            obs[done_id], _ = env.reset(done_id)
        for r, l in zip(*rec(reward, done_id)):
            res["test/reward"].append(r)
            res["test/len"].append(l)
            n_done += 1

    res["test/total_time"].append(time.time() - test_start_time)
    return res


def train(cfg: Conf) -> None:
    cfg.algo: AlgoConf
    seeder = Seeder(cfg.seed)

    env, test_env = cfg.env(seeder(), seeder())
    keygen = JRK(seeder())

    rb = cfg.replay(env, seed=seeder(), batch=cfg.env.num_envs)
    print(cfg.algo)
    algo = cfg.algo(keygen(), env)

    env.observation_space.seed(seeder())
    env.action_space.seed(seeder())

    updates = 0
    with closing(Logger(cfg.log, cfg)) as logger:
        obs, _ = env.reset()
        erec = Erec(obs.shape[0])
        for epoch in range(ceil(cfg.train.steps / cfg.train.steps_per_epoch)):
            if epoch % cfg.log.dump_every == cfg.log.dump_every - 1:
                logger.pickle(epoch, algo=algo.get_state())

            for step in range(
                epoch * cfg.train.steps_per_epoch,
                (1 + epoch) * cfg.train.steps_per_epoch,
            ):
                if step < cfg.train.random_period:
                    action = [env.action_space.sample() for _ in range(obs.shape[0])]
                    action = np.stack(action, axis=0)
                else:
                    action = algo.act(keygen(), obs)

                obs1, reward, term, trunc, _ = env.step(action)
                done = term | trunc
                done_id = np.where(done)[0]
                for r, l in zip(*erec(reward, done_id)):
                    logger.append({"train/reward": r, "train/len": l})

                rb(s0=obs, a=action, r=reward, s1=obs1, term=term)
                obs = obs1
                if len(done_id) > 0:
                    obs[done_id], _ = env.reset(done_id)

                if step >= cfg.train.start_training and rb.has_batch(
                    cfg.train.batch_size
                ):
                    for _ in range(cfg.train.replay_ratio):
                        aux = algo.update(keygen(), data=rb[cfg.train.batch_size])
                        updates += 1
                        logger.append(aux)
                        if (
                            cfg.train.reset_every is not None
                            and updates % cfg.train.reset_every == 0
                        ):
                            algo.reset(keygen())

            test_r = test_agent(cfg, algo, test_env)
            test_r["test/reward_std"].append(np.std(test_r["test/reward"]))
            logger.extend(test_r)
            logger(step)
