import os
from dataclasses import dataclass
from jax import nn, lax, vmap, jit
from jac.consts import jnp, dc
from jac.utils import wmap, set_plt
from math import prod
from math import log
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EscapeGrid:
    x: tuple
    s: int = dc.field(init=False)
    a: int = dc.field(init=False)
    n: int = dc.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "s", prod(self.x))
        object.__setattr__(self, "a", len(self.x) * 2)
        object.__setattr__(self, "n", len(self.x))

    def unflaten(self, x: int):
        """
        Convert from index to board state
        """
        y = []
        for i in self.x:
            x, m = jnp.divmod(x, i)
            y.append(m)
        return jnp.array(y)

    def flatten(self, x):
        """
        Convert from board state to index
        """
        a = [1]
        for i in self.x[:-1]:
            a.append(a[-1] * i)
        return (jnp.array(a) * x).sum()

    def array(self):
        return jnp.array(self.x)

    def oneencode(self, x):
        """
        One hot encode each of the elements of x, size is taken from grid
        """
        return jnp.concatenate(
            [nn.one_hot(x[..., idx], i) for idx, i in enumerate(self.x)], axis=-1
        )

    def step(self, s, a):
        """
        step the env, if the action is zero or we cannot change the state (i.e. invalid actions),
        we also stop.
        clipping makes sure that the state stays valid
        """
        assert s.ndim == a.ndim + 1

        s1 = wmap(
            lambda s, a: lax.cond(
                a >= self.n, lambda: s.at[a - self.n].add(-1), lambda: s.at[a].add(1)
            ),
            s.ndim - 1,
        )(s, a)
        s1 = jnp.clip(s1, 0, self.array() - 1)
        t = (s1 == self.array()).all(axis=-1)
        return s1, t

class F:
    def __init__(self, x, n, t=0.1, gamma=1):
        self.x = x
        self.n = n
        self.t = t
        self.gamma = gamma
        g = EscapeGrid((x,) * n)

        ns = vmap(
            lambda s: vmap(lambda a: g.flatten(g.step(g.unflaten(s), a)[0]))(
                jnp.arange(g.a)
            )
        )(jnp.arange(g.s))

        def f(x):
            Q, i, _ = x
            V = t * nn.logsumexp(Q / t, axis=-1)
            Q1 = -1 + gamma * wmap(lambda s: V[s], 2)(ns)
            Q1 = Q1.at[-1].set(0)
            conv = jnp.linalg.norm(Q1 - Q) < 1e-6
            return Q1, i + 1, conv

        def term(x):
            _, i, conv = x
            return ~conv & (i < 10000)

        Q, i, Q_conv = lax.while_loop(
            term, f, (jnp.zeros((g.s, g.a)), jnp.array(0), jnp.array(False))
        )
        self.Q = nn.logsumexp(Q[0], axis=-1)
        self.Q_conv = Q_conv.item()
        self.Q_iter = i.item()

        pi = nn.softmax(Q / t)
        self.pi = pi[0]
        def TTL(x):
            L, i, _ = x
            V = (pi * L).sum(-1)
            L1 = 1 + wmap(lambda s: V[s], 2)(ns)
            L1 = L1.at[-1].set(0)
            conv = jnp.linalg.norm(L1 - L) < 1e-6
            return L1, i + 1, conv
        L, i, L_conv = lax.while_loop(term, TTL, (jnp.zeros((g.s, g.a)), jnp.array(0), jnp.array(False)))
        self.L = (L[0] * pi[0]).sum()
        self.L_i = i.item()
        self.L_conv = L_conv.item()

if __name__ == '__main__':
    set_plt(True)
    base_temp = .4
    width = 4
    x = range(2, 10)
    y = [F(t=base_temp, x=width, n=i) for i in x]
    yn = [F(t=base_temp / log(2 * i), x=width, n=i) for i in x]        

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(x, [i.L for i, j in zip(y, x)], label="SQL")
    ax.plot(x, [i.L for i, j in zip(yn, x)], '--', label="Decoupled SQL")
    ax.plot(x, [(i.x - 1) * i.n for i, j in zip(yn, x)], '.', label="Shortest path")
    ax.set_ylim(0, 200)
    ax.legend()
    ax.set_ylabel("Expected episode length")
    ax.set_xlabel("Dimensionality")
    fig.savefig("ndim.pdf", bbox_inches="tight")