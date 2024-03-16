from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp
from copy import deepcopy


class ReplayBuffer(object):
    def __init__(self, capacity, warmup, priority, rng: np.random.Generator = np.random.default_rng()):
        self.capacity = capacity
        self.warmup = warmup
        self.priority = priority
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"
        self.shared_cap = mp.Value("q", 0)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.buffer: List[tuple] = []
        self.position = 0
        self.rng = rng

    def push_many(self, rewards, *args):
        args = deepcopy(args)  # move back from shared memory
        rewards = deepcopy(rewards)
        for i in range(len(rewards)):
            self._push(rewards[i], *[a[i] for a in args])

    def push(self, reward, *args):
        reward = deepcopy(reward)
        args = deepcopy(args)
        self._push(reward, *args)

    def _push(self, reward, *args):
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.rewards[self.position] = float(reward)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity
        self.shared_cap.value = len(self.buffer)

    def sample(self, batch_size):
        if self.priority:
            weights = self.rewards[: len(self.buffer)]
            weights = np.maximum(weights, 1e-5)
            weights = weights / weights.max()
            weights = weights / weights.sum()
        else:
            weights = None
        idxs = self.rng.choice(len(self.buffer), batch_size, p=weights)
        out = list(zip(*[self.buffer[idx] for idx in idxs]))
        for i in range(len(out)):  # pylint: disable=consider-using-enumerate
            # stack if all elements are numpy arrays or torch tensors
            # (this is much more efficient to send arrays through multiprocessing queues)
            if all((isinstance(x, np.ndarray) for x in out[i])):
                out[i] = np.stack(out[i], axis=0)
            elif all((isinstance(x, torch.Tensor) for x in out[i])):
                out[i] = torch.stack(out[i], dim=0)
        return tuple(out)
    
    def __len__(self):
        return len(self.buffer)
