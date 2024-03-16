from dataclasses import MISSING, dataclass
from typing import Optional
from jac.replay import Trans


class Algorithm:
    def update(self, key, data: Trans):
        raise NotImplementedError()

    def act(self, key, obs):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    def reset(self, key):
        raise NotADirectoryError()


@dataclass(eq=True, kw_only=True, unsafe_hash=True)
class AlgoConf:
    # net
    hidden_layers: int = MISSING
    hidden_units: int = MISSING
    activation: str = MISSING
    loss_fn: str = MISSING
    # setup
    temp: float = MISSING
    max_entropy: Optional[float] = MISSING
    min_entropy: Optional[float] = MISSING
    Rp: float = MISSING
    mellow: bool = MISSING
    decoupled: bool = MISSING
    discount: float = MISSING
    # sampling
    quad: bool = MISSING
    diag: bool = MISSING
    sampling_head: int = MISSING
    trian: bool = MISSING

    def __call__(self, key, env) -> Algorithm:
        raise NotImplementedError()
