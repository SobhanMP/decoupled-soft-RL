from dataclasses import MISSING, dataclass
from enum import Enum
from functools import partial, reduce
from math import prod
from typing import Any

import chex
import jax
import optax
from jax import grad, jit, lax

from jac.consts import dc, hk, jnp, jr, np, gym
from jac.algo import AlgoConf, Algorithm
from jac.nets import make_pi, make_Q, sample_action
from jac.replay import Trans
from jac.sampling import get_sampling, SamplingBase
from jac.utils import loss_fn, mse, tree_ema


class TempHueristic(Enum):
    SAC = "sac"
    UNIFORM = "uniform"
    ZERO = "zero"


@dataclass(eq=True, kw_only=True, unsafe_hash=True)
class SACConf(AlgoConf):
    # opt
    lr_Q: float = MISSING
    lr_pi: float = MISSING
    ema: float = MISSING
    backup_n: int = MISSING
    p_loss_n: int = MISSING
    t_loss_n: int = MISSING
    n_q: int = MISSING
    stop_gradient: bool = MISSING
    auto_temp: bool = MISSING
    heuristic_mult: float = MISSING
    temp_heuristic: TempHueristic = MISSING
    lr_lt: float

    def __call__(self, key, env):
        return SAC(key, self, env)


@dataclass(frozen=True)
class SACf:
    Q: hk.Transformed
    q_opt: optax.GradientTransformation
    pi: hk.Transformed
    p_opt: optax.GradientTransformation
    sampling: SamplingBase
    lt: float
    lt_opt: optax.GradientTransformation

    def __call__(self, key, obs, action):
        qkey, pkey = jr.split(key)
        q_params = self.Q.init(
            qkey,
            obs,
            action,
        )
        q_state = self.q_opt.init(q_params)

        p_params = self.pi.init(
            pkey,
            s=obs,
        )
        p_state = self.p_opt.init(p_params)
        temp_state = self.lt_opt.init(self.lt)
        lt = jnp.asarray([self.lt])
        return SACState(
            q=q_params,
            qt=q_params,
            qs=q_state,
            p=p_params,
            ps=p_state,
            lt=lt,
            lt_s=temp_state,
            t=jnp.exp(lt[0]),
        )  # type: ignore

    # https://github.com/deepmind/chex/issues/155


@chex.dataclass(frozen=True)
class SACState:
    q: dict
    qt: dict
    qs: dict
    p: dict
    ps: dict
    lt: jax.Array
    lt_s: Any
    t: jax.Array


class SAC(Algorithm):
    def __init__(self, key, cfg: SACConf, env: gym.Env):
        Q = make_Q(
            h=cfg.hidden_units,
            l=cfg.hidden_layers,
            activation=cfg.activation,
            n=cfg.n_q,
        )
        q_opt = optax.adam(cfg.lr_Q)

        action_nums = env.action_space.shape
        high = np.broadcast_to(env.action_space.high, action_nums)
        low = np.broadcast_to(env.action_space.low, action_nums)
        cfg.max_entropy = float(np.log(high - low).sum())
        cfg.min_entropy = float(np.log(1e-3) * sum(action_nums))
        if cfg.max_entropy <= cfg.min_entropy:
            cfg.min_entropy = cfg.max_entropy - 3
        temp = cfg.temp
        if cfg.decoupled:
            temp /= float(cfg.max_entropy - cfg.min_entropy)

        sampling = get_sampling(
            env.action_space.shape[-1],
            diag=cfg.diag,
            trian=cfg.trian,
            quad=cfg.quad,
            low=env.action_space.low,
            high=env.action_space.high,
            n=cfg.sampling_head,
        )

        pi = make_pi(
            h=cfg.hidden_units,
            l=cfg.hidden_layers,
            activation=cfg.activation,
            sampling=sampling,
        )
        p_opt = optax.adam(cfg.lr_pi)
        lt = jnp.log(np.array(temp))
        lt_opt = optax.adam(cfg.lr_lt)
        self.obs = env.observation_space.sample().astype(np.float32)
        self.action = env.action_space.sample().astype(np.float32)

        self.sf = SACf(
            Q=Q,
            q_opt=q_opt,
            pi=pi,
            p_opt=p_opt,
            sampling=sampling,
            lt=lt.item(),
            lt_opt=lt_opt,
        )
        self.state = self.reset(key)
        self.cfg = cfg

    def update(self, key, data: Trans):
        self.state, aux = sac_update(key, self.cfg, self.sf, self.state, data)
        return aux

    def act(self, key, obs):
        return sample_action(
            key,
            sampling=self.sf.sampling,
            net=self.sf.pi,
            params=self.state.p,
            obs=obs,
            shape=(),
        )

    def get_state(self):
        return self.state

    def reset(self, key):
        self.state = self.sf(key, self.obs, self.action)
        return self.state


def backup(key, cfg: SACConf, sf: SACf, state: SACState, data: Trans):
    qp = state.qt if state.qt is not None else state.q
    sp = sf.pi.apply(state.p, data.s[:, 1:])
    a, lp = sf.sampling(key, sp, shape=(cfg.backup_n,))
    qs = sf.Q.apply(qp, data.s[:, 1:, None], a)
    q = reduce(jnp.minimum, qs)
    v = q - state.t * lp
    v = v.mean(-1)  # remvoe backup_n
    if cfg.mellow:
        v = v - cfg.max_entropy * state.t
    y = data.r + cfg.Rp + cfg.discount * ~data.t * v

    return y, v


def q_loss(key, cfg: SACConf, sf: SACf, state: SACState, data: Trans):
    y, v = backup(key, cfg, sf, state, data)
    if cfg.stop_gradient:
        y = lax.stop_gradient(y)  # just to be safe
    qs = sf.Q.apply(state.q, data.s[:, :-1], data.a)
    qlosses = [loss_fn(cfg.loss_fn)(q, y, mask=data.mask, agg="mean") for q in qs]
    aux = {"vmean": v.mean()}

    for i, ql in enumerate(qlosses):
        aux["qloss" + str(i)] = ql

    return sum(qlosses), aux


def q_update(key, cfg: SACConf, sf: SACf, state: SACState, data: Trans):
    q_g, q_aux = grad(
        lambda q: q_loss(key, cfg, sf, dc.replace(state, q=q), data),
        has_aux=True,
    )(state.q)
    aux = {}
    aux["|qg|"] = jnp.linalg.norm(jax.flatten_util.ravel_pytree(q_g)[0])
    q_updates, q_state = sf.q_opt.update(q_g, state.qs)
    q_params = optax.apply_updates(state.q, q_updates)

    return dc.replace(state, q=q_params, qs=q_state), aux | q_aux


def p_loss(key, cfg: SACConf, sf: SACf, state: SACState, data: Trans):
    sp = sf.pi.apply(state.p, data.s[..., :-1, :])
    a, lp = sf.sampling(key, sp, shape=(cfg.p_loss_n,))
    qs = sf.Q.apply(state.q, data.s[..., :-1, None, :], a)
    q = reduce(jnp.minimum, qs)
    ploss = ((state.t * lp - q) * data.mask / data.mask.sum()).sum()
    return ploss, {"ploss": ploss}


def p_update(key, cfg: SACConf, sf: SACf, state: SACState, data: Trans):
    pi_g, aux = grad(
        lambda p: p_loss(key, cfg, sf, dc.replace(state, p=p), data),
        has_aux=True,
    )(state.p)
    aux["|pg|"] = jnp.linalg.norm(jax.flatten_util.ravel_pytree(pi_g)[0])

    p_updates, p_state = sf.p_opt.update(pi_g, state.ps)
    p_params = optax.apply_updates(state.p, p_updates)

    return dc.replace(state, p=p_params, ps=p_state), aux


def lt_loss(key, cfg: SACConf, sf: SACf, state: SACState, data: Trans):
    assert state.lt.shape == (1,)
    t = jnp.exp(state.lt[0])
    sp = sf.pi.apply(state.p, data.s[..., :-1, :])
    _, lp = sf.sampling(key, sp, shape=(cfg.t_loss_n,))
    if cfg.temp_heuristic == TempHueristic.SAC:
        H = -prod(data.a.shape[1:]) * cfg.heuristic_mult
    elif cfg.temp_heuristic == TempHueristic.UNIFORM:
        # H(pi) - min_H >= a * max_H - min_H
        # H(pi) >= a * max_H + (1 - a) * min_H
        H = (
            cfg.max_entropy * cfg.heuristic_mult
            + (1 - cfg.heuristic_mult) * cfg.min_entropy
        )
    elif cfg.temp_heuristic == TempHueristic.ZERO:
        H = 0
    else:
        raise NotImplementedError
    tloss = (-lp.mean(-1) - H).mean()
    aux = {"tloss": tloss}
    return t * tloss, aux


def lt_update(key, cfg: SACConf, sf: SACf, state: SACState, data: Trans):
    lt_g, aux = grad(
        lambda lt: lt_loss(key, cfg, sf, dc.replace(state, lt=lt), data),
        has_aux=True,
    )(state.lt)
    aux["|ltg|"] = jnp.linalg.norm(jax.flatten_util.ravel_pytree(lt_g)[0])

    lt_updates, lt_state = sf.lt_opt.update(lt_g, state.lt_s)
    lt_params = optax.apply_updates(state.lt, lt_updates)
    lt_params = jnp.clip(lt_params, -30, 30)  # numerical stability
    return dc.replace(state, lt=lt_params, lt_s=lt_state), aux


@partial(jit, static_argnames=["sf", "cfg"])
def sac_update(key, cfg: SACConf, sf: SACf, state: SACState, data: Trans):
    qkey, pkey, tkey = jr.split(key, 3)
    state, q_aux = q_update(qkey, cfg, sf, state, data)
    state, p_aux = p_update(pkey, cfg, sf, state, data)
    if cfg.auto_temp:
        state, lt_aux = lt_update(tkey, cfg, sf, state, data)
        state = dc.replace(state, t=jnp.exp(state.lt[0]))

        assert state.t.shape == ()
    else:
        lt_aux = {}

    lt_aux["temp"] = state.t
    q_target = tree_ema(ema=cfg.ema, target=state.qt, params=state.q)

    return dc.replace(state, qt=q_target), q_aux | p_aux | lt_aux
