from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from math import sqrt
import time
import logging
from omegaconf import OmegaConf
import jax
from jax import jit, vmap
from jax.tree_util import tree_map
from matplotlib import pyplot as plt
import seaborn as sns

from jac.consts import jr, jnp, np

__all__ = ["tree_ema", "classify", "wmap", "JRK", "mse"]


@partial(jit, static_argnames=["ema"])
def tree_ema(ema, target, params):
    return tree_map(lambda x, y: x * ema + y * (1 - ema), target, params)


def mask_apply(x, mask, agg):
    if mask is not None:
        x = x * mask
    match agg:
        case "none":
            pass
        case "mean":
            if mask is not None:
                x = (x / mask.sum()).sum()
            else:
                x = x.mean()
        case "sum":
            x = x.sum()
        case _:
            raise ValueError(f"Unknown agg {agg}")
    return x


def mse(x, y=None, mask=None, agg="none"):
    if y is not None:
        x = x - y
    x = x * x
    return mask_apply(x, mask, agg)


def hubber(x, y=None, mask=None, agg="none", delta=1):
    if y is not None:
        x = x - y
    x = jnp.where(
        jnp.abs(x) < delta,
        0.5 * x * x,
        delta * (jnp.abs(x) - 0.5 * delta),
    )
    return mask_apply(x, mask, agg)


def mae(x, y=None, mask=None, agg="none"):
    if y is not None:
        x = x - y
    x = abs(x)
    return mask_apply(x, mask, agg)


def loss_fn(name):
    return {"mse": mse, "hubber": hubber, "mae": mae}[name]


def linex(x):
    return jnp.exp(x) - x - 1


def stable_linex(x, y=None, lim=1, mult=1):
    "stable function trick for tensorflow, the second jnp.where makes sure the gradients exist"
    if y is not None:
        x = x - y
    x = x / mult
    g = jnp.exp(lim) - 1
    y = jnp.where(x > lim, lim, x)
    return jnp.where(x > lim, g * (x - lim) + linex(lim), linex(y))


def classify(l):
    def _classify(x: str):
        for i in l:
            if x.startswith(i):
                return i

    def g(x):
        h = {}
        for k, v in x.items():
            y = _classify(k)
            if y:
                h[k] = y
            elif isinstance(v, dict):
                h[k] = g(v)
            else:
                raise ValueError(f"Unknown key {k}")
        return h

    return g


def wmap(f, n=1):
    for _ in range(n):
        f = vmap(f)
    return f


class JRK:
    """
    Key generator class, calling the `()` gives a new key
    """

    def __init__(self, seed) -> None:
        self.key = jr.PRNGKey(seed)

    def __call__(self):
        self.key, sub = jr.split(self.key, 2)
        return sub


def vec_tril(x):
    n = x.shape[-1]
    n = int((sqrt(8 * n + 1) - 1) / 2 + 0.5)
    i, j = jnp.tril_indices(n)
    return jnp.zeros((*x.shape[:-1], n, n), dtype=x.dtype).at[..., i, j].set(x)


def select(x: jnp.ndarray, y: jax.Array, trailing=0):
    """
    select the elements of the last dim of x according to y
    """
    assert x.ndim == y.ndim + 1 + trailing
    return wmap(lambda x, y: x[y], y.ndim)(x, y)


def kl(r, rho):
    return jnp.where(r == 0, 0.0, r * jnp.log(r / rho)).sum(-1).mean()


def H(pi):
    return -jnp.where(pi == 0, 0.0, pi * jnp.log(pi)).sum(-1)


def plot_n_heatmap(ns, axes, shape=None):
    """
    plot n heatmaps next on the axes with the same coloring
    """
    vmin = min(map(jnp.min, ns))
    vmax = max(map(jnp.max, ns))
    ims = []
    for i, j in zip(ns, axes):
        if shape is not None:
            i = i.reshape(shape)
        ims.append(j.imshow(i, vmin=vmin, vmax=vmax, origin="lower"))
    return ims


def set_plt(usetex=False):
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (4, 3)
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"

    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.size": 11,
            "font.family": "serif",
            # "text.latex.unicode": True,
        }
    )


class Erec:
    def __init__(self, n):
        self.r = np.zeros(n)
        self.l = np.zeros(n, dtype=np.int32)

    def __call__(self, r, done_id):
        self.r += r
        self.l += 1
        res = ([self.r[i] for i in done_id], [self.l[i] for i in done_id])
        self.r[done_id] = 0
        self.l[done_id] = 0
        return res


def dict_flatten(x):
    y = {}
    for k, v in x.items():
        if isinstance(v, dict):
            v = dict_flatten(v)
            for k2, v2 in v.items():
                y[k + "." + k2] = v2
        else:
            y[k] = v
    return y


def demote(x):
    if x == np.float64:
        return np.float32
    elif x == np.int64:
        return np.int32
    elif x == int:
        return np.int32
    return x


class Seeder:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        return int(self.rng.integers(11111, 99999))
