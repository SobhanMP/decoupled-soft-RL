from dataclasses import MISSING, dataclass

from jac.utils import demote
from jac.replay import RBConf, Trans
from jac.consts import np


@dataclass(eq=True, kw_only=True)
class TransRBConf(RBConf):
    size: int = MISSING

    def __call__(self, env, seed, batch):
        return TransReplayBuffer(
            env.observation_space.sample(),
            env.action_space.sample(),
            seed,
            self.size,
        )


class TransReplayBuffer:
    def __init__(self, obs, action, seed, size) -> None:
        self.s0 = np.ndarray(
            (size, *obs.shape),
            dtype=demote(obs.dtype),
        )
        self.s1 = np.ndarray(
            (size, *obs.shape),
            dtype=demote(obs.dtype),
        )
        self.a = np.ndarray((size, *action.shape), dtype=demote(action.dtype))
        self.r = np.ndarray((size,), dtype=np.float32)
        self.term = np.ndarray((size,), dtype=bool)
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.i = 0
        self.used = 0

    def __getitem__(self, batch_size):
        idx = self.rng.choice(
            self.used, size=batch_size
        )  # TODO, should we use unique choices?
        return Trans(  # b t *s
            s=np.stack([self.s0[idx], self.s1[idx]], axis=1),
            a=self.a[idx][:, None],
            r=self.r[idx][:, None],
            t=self.term[idx][:, None],
            mask=np.full((batch_size, 1), True, dtype=bool),
        )

    def has_batch(self, batch_size):
        return self.used >= batch_size

    def __call__(self, s0, a, r, s1, term):
        assert s0.shape[0] == s1.shape[0] == a.shape[0] == r.shape[0] == term.shape[0]
        b = s0.shape[0]
        i0 = self.i
        im = min(self.size, self.i + b)
        self.i = im % self.size
        self.used = max(self.used, im)

        l = im - i0
        self.s0[i0:im] = s0[:l]
        self.a[i0:im] = a[:l]
        self.r[i0:im] = r[:l]
        self.s1[i0:im] = s1[:l]
        self.term[i0:im] = term[:l]

        if l < b:
            self(s0[l:], a[l:], r[l:], s1[l:], term[l:])
