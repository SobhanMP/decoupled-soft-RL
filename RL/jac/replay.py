from dataclasses import MISSING, dataclass
from math import ceil
from typing import Any

import chex

from jac.consts import np, gym
from jac.utils import demote


class ReplayBuffer:
    def seed(self, seed):
        raise NotImplementedError

    def has_batch(self, batch_size):
        raise NotImplementedError

    def __getitem__(self, batch_size):
        raise NotImplementedError


@dataclass(eq=True, kw_only=True)
class RBConf:
    def __call__(self, env: gym.Env, seed):
        raise NotImplementedError


@chex.dataclass
class Trans:
    s: chex.Array
    a: chex.Array
    r: chex.Array
    t: chex.Array
    mask: chex.Array

