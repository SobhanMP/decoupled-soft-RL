from jac.consts import jnp
from jac.utils import wmap


# mask of the valid element of st_n
def sub_tb_w(discount, mask):
    h = mask.shape[-1]
    x = jnp.full((h, h), discount)
    if discount != 1:
        x = jnp.cumprod(x, axis=-1)

    x = jnp.where(mask[:, None] & mask[None], x, 0.0)
    return jnp.triu(x)


def cross_cumsum(x):
    cx = jnp.cumsum(x, 0)  # start from zero
    cx1 = jnp.roll(cx, 1, axis=0).at[0].set(0)
    return cx[None] - cx1[:, None]


def cross_diff(x, r):
    if r is None:
        x, x1 = x[:-1], x[1:]
    else:
        x1 = jnp.roll(x, -1, axis=0).at[-1].set(r)

    return x[:, None] - x1[None]


def sub_tb(x, v, r, mask=None):
    """
    Calculate the sub-TB objective with DP.
    ret[i,j] = v[i] - v[j+1] + sum(x[i:j+1]),
    except when mask[j+1] is False, then r is used instead of
    """
    cx = cross_cumsum(x)
    if mask is not None and r is not None:
        v = jnp.where(mask, v, r)
    cv = cross_diff(v, r)
    assert cv.shape[0] == cx.shape[0]
    ret = cv + cx
    if mask is not None:
        ret = ret * (mask[:, None] & mask[None])
    return jnp.triu(ret)


def sub_tb_n(pn, n, mask=None, discount=None):
    m = mask[1:][::-1]
    l = sub_tb(pn[1:][::-1], n[1:][::-1], 0, mask=m)
    if discount is not None:
        w = sub_tb_w(discount, m)
        return l, w

    return l


def sub_ft(x, v, r, mask=None):
    assert x.ndim == 1
    assert v.ndim == 1
    if mask is not None:
        x = x * mask

    cx = jnp.cumsum(x[::-1])[::-1]
    if r is None:
        v, r = v[:-1], v[-1]
    y = v + cx - r
    if mask is not None:
        y = y * mask
    return y


def sub_ft_w(discount, mask):
    assert mask.ndim == 1
    h = mask.shape[-1]
    x = jnp.full(h, discount)
    if discount != 1:
        x = jnp.cumprod(x, axis=-1)[::-1]
    return x * mask


def sub_ft_n(pn, n, mask=None, discount=None):
    m = mask[1:][::-1]
    l = sub_ft(pn[1:][::-1], n[1:][::-1], 0, mask=m)
    if discount is not None:
        w = sub_ft_w(discount, m)
        return l, w
    return l


def sub_it(x, v, r, mask=None):
    assert x.ndim == v.ndim == 1
    if r is not None:
        if mask is not None:
            v = jnp.where(mask, v, r)
        v = jnp.concatenate([v, jnp.array(r).reshape((1,))])

    if mask is not None:
        x = x * mask
        v0 = v[jnp.argmax(mask)]
    else:
        v0 = v[0]
    cx = jnp.cumsum(x)
    assert v.shape[0] == x.shape[0] + 1
    res = v0 + cx - v[1:]
    if mask is not None:
        res = res * mask
    return res


def sub_it_w(discount, mask):
    assert mask.ndim == 1
    h = mask.shape[-1]
    x = jnp.full(h, discount)
    if discount != 1:
        x = jnp.cumprod(x, axis=-1)
    return x * mask


def sub_it_n(pn, n, mask=None, discount=None):
    m = mask[1:][::-1]
    l = sub_it(pn[1:][::-1], n[1:][::-1], 0, mask=m)
    if discount is not None:
        w = sub_it_w(discount, m)
        return l, w
    return l


def pcl(x, v, gamma):
    h = x.shape[-1]
    assert x.shape[:-1] == v.shape[:-1]
    assert v.shape[-1] == h + 1
    d = jnp.cumprod(jnp.full((h,), gamma))
    v = v.at[:, :-1].mul(d).at[:, -1].mul(d[-1] * gamma)
    x = x * d
    l = wmap(sub_tb, x.ndim - 1)(x, v[:, :-1], v[:, -1])
    l = l / d[:, None]
    return l
