from dataclasses import dataclass
from jax import nn

from jac.consts import jnp


@dataclass
class Aggregation:
    def __call__(self, x):
        raise NotImplementedError


@dataclass
class LogSumExp(Aggregation):
    t: float

    def __call__(self, x):
        return self.t * nn.logsumexp(x / self.t, -1)


@dataclass
class Mellow(Aggregation):
    parent: Aggregation

    def __call__(self, x):
        return self.parent(x) - self.parent(jnp.zeros_like(x))


@dataclass
class Decoupled(Aggregation):
    parent: Aggregation

    def __call__(self, x):
        return self.parent(x) / self.parent(jnp.zeros_like(x))
