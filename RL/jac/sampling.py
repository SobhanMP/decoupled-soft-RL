from jax import nn, vmap

from jac.utils import vec_tril, select

from .consts import MAX_LS, MIN_LS, jnp, jr

__all__ = ["SamplingBase", "DiagNormal", "Normal", "TanhSampling", "MMsampling"]


class SamplingBase:
    def param_count(self):
        raise NotImplementedError

    def __call__(self, key, x, shape):
        raise NotImplementedError

    def lp(self, x, y):
        raise NotImplementedError()


class DiagNormal(SamplingBase):
    def __init__(self, k, quad, min_mu=None, max_mu=None) -> None:
        assert k >= 1
        self.k = k
        self.quad = quad
        self.min_mu = min_mu
        self.max_mu = max_mu

    def param_count(self):
        return self.k * 2

    @classmethod
    def get_diag_s(cls, quad, x):
        if quad:
            min_s = jnp.exp(MIN_LS)
            s = min_s + x * x
            ls = jnp.log(s)
        else:
            x = MIN_LS + 0.5 * (MAX_LS - MIN_LS) * (jnp.tanh(x) + 1)
            ls = x
            s = jnp.exp(x)
        return s, ls

    def get_param(self, x):
        assert x.shape[-1] == self.param_count()
        mu = x[..., : self.k]
        if self.min_mu is not None:
            mu = self.min_mu + 0.5 * (self.max_mu - self.min_mu) * (jnp.tanh(mu) + 1)
        s, ls = self.get_diag_s(self.quad, x[..., self.k :])
        return mu, s, ls

    def lp_z(self, ldet, zsq):
        return -self.k * jnp.log(2 * jnp.pi) / 2 - ldet - zsq / 2

    def lp(self, x, y):
        mu, s, ls = self.get_param(x)
        assert mu.ndim == y.ndim
        z = (y - mu) / s
        zsq = jnp.sum(z * z, axis=-1)
        ldet = jnp.sum(ls, axis=-1)
        return self.lp_z(ldet, zsq)

    def __call__(self, key, x, shape):
        mu, s, ls = self.get_param(x)
        sparse_shape = (*x.shape[:-1], *([1] * len(shape)), self.k)
        dense_shape = (*x.shape[:-1], *shape, self.k)

        mu = mu.reshape(sparse_shape)
        s = s.reshape(sparse_shape)
        ls = ls.reshape(sparse_shape)

        x = mu

        """
        if key is none, no noise is used
        """
        if key is not None:
            z = jr.normal(key, dense_shape)
            x = x + s * z
            zsq = jnp.sum(z * z, -1)
        else:
            zsq = 0
        ldet = jnp.sum(ls, axis=-1)
        return x, self.lp_z(ldet, zsq)


class Normal(DiagNormal):
    def __init__(self, k) -> None:
        assert k >= 1
        self.k = k

    def param_count(self):
        return self.k * (self.k + 1)

    @staticmethod
    def get_full_s(x):
        return jnp.eye(x.shape[-1]) * jnp.exp(MIN_LS) + jnp.matmul(
            jnp.swapaxes(x, -1, -2), x
        )

    def get_param(self, x):
        assert x.shape[-1] == self.k * (self.k + 1)
        mu = x[..., : self.k]
        s = self.get_full_s(x[..., self.k :].reshape((*x.shape[:-1], self.k, self.k)))
        return mu, s

    @staticmethod
    def cldet(l):
        l_diag = jnp.diagonal(l, axis1=-2, axis2=-1)
        return jnp.sum(
            jnp.log(l_diag),
            axis=-1,
        )

    def lp(self, x, y):
        mu, s = self.get_param(x)
        assert mu.shape[-1] == y.shape[-1]
        l = jnp.linalg.cholesky(s)
        z = jnp.linalg.solve(l, y - mu)
        zsq = jnp.sum(z * z, axis=-1)

        return self.lp_z(self.cldet(l), zsq)

    def __call__(self, key, x, shape):
        """
        if key is none, no noise is used
        """
        T = x.shape[:-1]
        mu, s = self.get_param(x)
        l = jnp.linalg.cholesky(s)
        return self.f(key, T, mu, l, shape)

    def f(self, key, T, mu, l, shape):
        dense_shape = (*T, *shape)
        sparse_shape = (*T, *([1] * len(shape)))

        x = mu.reshape(*sparse_shape, self.k)
        l = l.reshape(*sparse_shape, self.k, self.k)

        if key is not None:
            z = jr.normal(key, (*dense_shape, self.k))

            sd = jnp.einsum("...ij,...j->...i", l, z)
            x = x + sd
            zsq = jnp.sum(z * z, axis=-1)
        else:
            zsq = 0
        lp = self.lp_z(self.cldet(l), zsq)
        return x, lp


class TriagNormal(Normal):
    """
    Predict the factorization directly, should be the fastest, most precise but potentially wrong
    """

    def __init__(self, k) -> None:
        self.k = k

    def param_count(self):
        return self.k + ((self.k + 1) * self.k) // 2

    def get_param(self, x):
        assert x.shape[-1] == self.param_count()
        mu = x[..., : self.k]
        l = self.get_l(x[..., self.k :])
        return mu, l

    @staticmethod
    def get_l(x):
        x = vec_tril(x)
        y = jnp.diagonal(x, axis1=-2, axis2=-1)
        y = y * y + jnp.exp(MIN_LS / 2)
        a = jnp.arange(x.shape[-1])
        return x.at[..., a, a].set(y)

    def __call__(self, key, x, shape):
        """
        if key is none, no noise is used
        """
        T = x.shape[:-1]
        mu, l = self.get_param(x)
        return self.f(key, T, mu, l, shape)

    def lp(self, x, y):
        mu, l = self.get_param(x)
        assert mu.shape[-1] == y.shape[-1]
        z = jnp.linalg.solve(l, y - mu)
        zsq = jnp.sum(z * z, axis=-1)

        return self.lp_z(self.cldet(l), zsq)


class TanhSampling(SamplingBase):
    def __init__(self, parent, low, high):
        self.parent = parent
        assert len(low) >= 1
        assert len(high) >= 1
        low = jnp.array(low)
        high = jnp.array(high)
        self.scale = jnp.array((high - low) / 2)
        self.disp = jnp.array((high + low) / 2)

    def __call__(self, key, x, shape):
        x, lp = self.parent(key, x, shape)
        return self.transform(x, lp)

    def param_count(self):
        return self.parent.param_count()

    @classmethod
    def ldetJ(cls, x):
        x = 2 * nn.softplus(2 * x) - jnp.log(4) - 2 * x
        return jnp.sum(x, -1)

    @classmethod
    def ldetJ_arctanh(cls, x):
        """
        For testing, ldetJ(x) = ldetJ_arctanh(tanh(x)) but ldetJ is more stable
        """
        # diagonal of jacobian of arctanh
        x = 1 - x * x
        return -jnp.sum(jnp.log(x), -1)

    def transform(self, x, lp):
        ldet = self.ldetJ(x)
        a = self.scale * jnp.tanh(x) + self.disp
        return a, lp + ldet - jnp.log(self.scale).sum(-1)

    def lp(self, x, y):
        z = (y - self.disp) / self.scale
        z = jnp.clip(z, -0.99999, 0.99999)  # TODO REMOVE
        z = jnp.arctanh(z)

        lp = self.parent.lp(x, z)
        ldet = self.ldetJ(x)
        return lp + ldet - jnp.log(self.scale).sum(-1)


class MMsampling(SamplingBase):
    def __init__(self, parent: SamplingBase, n) -> None:
        self.parent = parent
        self.n = n

    def param_count(self):
        return self.n * (1 + self.parent.param_count())

    def f(self, x, shape):
        T = x.shape[:-1]
        S = shape
        # x: *t, _
        # p: *t, n mixture params
        # q: *t, n, _  parent params
        # qs: *t, *s, n, F parent samples
        # qp: *t, *s, n parent lp
        # pp: *t, *s mixture lp
        # qsf: *t, *s, F
        # qpf: *t, *s
        p, q = x[..., : self.n], x[..., self.n :]
        q = q.reshape((*T, self.n, self.parent.param_count()))
        sparse_shape = (*T, *([1] * len(S)), p.shape[-1])
        dense_shape = (*T, *S, p.shape[-1])
        return p, q, T, S, sparse_shape, dense_shape

    def __call__(self, key, x, shape):
        p, q, T, S, sparse_shape, dense_shape = self.f(x, shape)
        if key is None:
            pp = nn.log_softmax(p, axis=-1)
            qs, qp = vmap(lambda x: self.parent(None, x, shape))(q)
            pind = jnp.argmax(pp + qp, axis=-1)  # heuristic
        else:
            key, subkey = jr.split(key, 2)
            pind = jr.categorical(
                key,
                jnp.broadcast_to(p.reshape(sparse_shape), dense_shape),
                shape=(*T, *S),
            )

            subkey = jr.split(subkey, self.n)
            qs, qp = vmap(
                lambda k, x: self.parent(k, x, shape),
                in_axes=(0, -2),
                out_axes=(-2, -1),
            )(subkey, q)

        qsf = select(qs, pind, trailing=1)

        return qsf, self.lp(x.reshape((*sparse_shape[:-1], -1)), qsf)

    def lp(self, x, y):
        p, q = self.f(x, ())[:2]
        lp = vmap(lambda x: self.parent.lp(x, y), in_axes=-2, out_axes=-1)(q)
        pp = nn.log_softmax(p, axis=-1)
        return nn.logsumexp(pp + lp, -1)


def get_sampling(k, diag, trian, quad, low, high, n):
    if diag:
        assert not trian
    if not quad and not diag:
        raise ValueError("Cannot use diag without quad")
    if diag:
        x = DiagNormal(k, quad)
    elif trian:
        x = TriagNormal(k)
    else:
        x = Normal(k)

    if n > 1:
        x = MMsampling(x, n)

    return TanhSampling(x, low, high)
