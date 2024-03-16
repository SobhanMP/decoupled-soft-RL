from typing import Optional
from jax import nn
from jac.consts import hk, jnp


class Head(hk.Module):
    def __init__(self, out, width, name=None):
        super().__init__(name)
        self.l1 = hk.Linear(512 * width)
        self.out = hk.Linear(out)

    def __call__(self, x):
        return self.out(nn.relu(self.l1(x)))


class Encoder(hk.Module):
    def __init__(self, width, name: Optional[str] = None):
        super().__init__(name)
        self.width = width

    def __call__(self, x):
        T = x.shape[:-3]
        F = x.shape[-3:]
        x = x.reshape(-1, *F)
        x = nn.relu(hk.Conv2D(self.width * 32, 8, 4)(x / 255))
        x = nn.relu(hk.Conv2D(self.width * 64, 4, 2)(x))
        c = nn.relu(hk.Conv2D(self.width * 64, 4, 2)(x))
        return c.reshape(*T, *c.shape[1:])


def make_net(width, actions):
    @hk.without_apply_rng
    @hk.transform
    def f(x):
        c = Encoder(width, "e")(x)
        cf = c.reshape(*c.shape[:-3], -1)
        q = Head(actions, width, "q")(cf)

        return q

    return f
