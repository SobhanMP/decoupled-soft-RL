from functools import partial


from jax import jit, nn

from .consts import jnp, hk
from .sampling import SamplingBase

__all__ = ["make_Q", "make_pi"]


def activation_fn(x):
    if isinstance(x, str):
        return {
            "relu": nn.relu,
            "leaky": nn.leaky_relu,
            "elu": nn.elu,
            "selu": nn.selu,
        }[x]
    else:
        return x


def make_Q(h, l, activation, n):
    @hk.without_apply_rng
    @hk.transform
    def Q(s, a):
        x = jnp.concatenate([s, a], axis=-1)
        return [
            hk.nets.MLP(
                [*([h] * l), 1],
                name=f"Q{i}",
                activation=activation_fn(activation),
            )(x)[..., 0]
            for i in range(n)
        ]

    return Q


def make_V(h, l, activation, n):
    @hk.without_apply_rng
    @hk.transform
    def V(s):
        x = [
            hk.nets.MLP(
                [*([h] * l), 1],
                name=f"V{i}",
                activation=activation_fn(activation),
            )(s)[..., 0]
            for i in range(n)
        ]
        return x

    return V


def make_pi(h, l, activation, sampling: SamplingBase):
    @hk.without_apply_rng
    @hk.transform
    def pi(s):
        x = x = hk.nets.MLP(
            [h] * l,
            activation=activation_fn(activation),
            name="pi",
            activate_final=True,
        )(s)

        x = hk.Linear(sampling.param_count())(x)
        return x

    return pi


@partial(jit, static_argnames=["net", "shape", "sampling"])
def sample_action(key, sampling, net, params, obs, shape=()):
    sp = net.apply(params, obs)
    a, _ = sample = sampling(key, sp, shape=shape)
    return a
