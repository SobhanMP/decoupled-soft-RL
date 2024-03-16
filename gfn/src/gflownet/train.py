from collections import defaultdict, deque
import time
import json
from math import ceil
from torch.multiprocessing import Value
import os
import pathlib
import socket
from typing import Any, Callable, Dict, List, NamedTuple, NewType, Optional, Tuple
import numpy as np

from rdkit.Chem.rdchem import Mol as RDMol

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from gflownet.data.sampling_iterator import SamplingIterator
from gflownet.envs.graph_building_env import (
    GraphActionCategorical,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)
from gflownet.utils.misc import create_logger
from gflownet.utils.multiprocessing_proxy import mp_object_wrapper
from gflownet.utils.multiobjective_hooks import RewardStats
from gflownet.data.replay import ReplayBuffer

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", Tensor)  # type: ignore
byte_to_gb = 1024 * 1024 * 1024


class GFNAlgorithm:
    def compute_batch_losses(
        self, model: nn.Module, batch: gd.Batch, num_bootstrap: Optional[int] = 0
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Computes the loss for a batch of data, and proves logging informations
        Parameters
        ----------
        model: nn.Module
            The model being trained or evaluated
        batch: gd.Batch
            A batch of graphs
        num_bootstrap: Optional[int]
            The number of trajectories with reward targets in the batch (if applicable).
        Returns
        -------
        loss: Tensor
            The loss for that batch
        info: Dict[str, Tensor]
            Logged information about model predictions.
        """
        raise NotImplementedError()


class GFNTask:
    def cond_info_to_logreward(
        self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards
    ) -> RewardScalar:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.

        Parameters
        ----------
        cond_info: Dict[str, Tensor]
            A dictionary with various conditional informations (e.g. temperature)
        flat_reward: FlatRewards
            A 2d tensor where each row represents a series of flat rewards.

        Returns
        -------
        reward: RewardScalar
            A 1d tensor, a scalar log-reward for each minibatch entry.
        """
        raise NotImplementedError()

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        """Compute the flat rewards of mols according the the tasks' proxies

        Parameters
        ----------
        mols: List[RDMol]
            A list of RDKit molecules.
        Returns
        -------
        reward: FlatRewards
            A 2d tensor, a vector of scalar reward for valid each molecule.
        is_valid: Tensor
            A 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()


class GFNTrainer:
    def __init__(self, hps: Dict[str, Any], device: torch.device):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        hps: Dict[str, Any]
            A dictionary of hyperparameters. These override default values obtained by the `default_hps` method.
        device: torch.device
            The torch device of the main worker.
        """
        self.keepalive = []
        # self.setup should at least set these up:
        self.training_data: Dataset
        self.test_data: Dataset
        self.model: nn.Module
        # `sampling_model` is used by the data workers to sample new objects from the model. Can be
        # the same as `model`.
        self.sampling_model: nn.Module
        self.mb_size: int
        self.env: GraphBuildingEnv
        self.ctx: GraphBuildingEnvContext
        self.task: GFNTask
        self.algo: GFNAlgorithm
        # Override default hyperparameters with the constructor arguments
        self.hps = {**self.default_hps(), **hps}

        self.replay = self.hps["replay"]
        seed = self.hps["seed"]
        if seed > 0:
            torch.manual_seed(seed)
            np.random.seed(seed + 1)

        self.device = device
        # The number of processes spawned to sample object and do CPU work
        self.num_workers: int = self.hps["num_data_loader_workers"]
        # The ratio of samples drawn from `self.training_data` during training. The rest is drawn from
        # `self.sampling_model`.
        self.offline_ratio = self.hps["offline_ratio"]

        # idem, but from `self.test_data` during validation.
        self.valid_offline_ratio = 1
        # If True, print messages during training
        self.verbose = False
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = [RewardStats()]
        self.valid_sampling_hooks: List[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False
        # Pickle messages to reduce load on shared memory (conversely, increases load on CPU)
        self.pickle_messages = self.hps["mp_pickle_messages"]

        self.wandb = self.hps["wandb"]
        if self.wandb:
            import wandb

            wandb.init(entity="gnncan-t", project="gfn2", config=self.hps)
        self.rp = Value("d", 0.0)
        self.set_rp(0)

        if self.hps["replay"]:
            self.replay_buffer = ReplayBuffer(
                capacity=self.hps["replay_capacity"],
                warmup=self.hps["replay_warmup"],
                priority=self.hps["replay_priority"],
            )

        else:
            self.replay_buffer = None

        self.setup()

    def default_hps(self) -> Dict[str, Any]:
        return {
            "mp_pickle_messages": False,
            "num_data_loader_workers": 0,
            "offline_ratio": 0.5,
            "replay_priority": True,
            "replay_capacity": 10_000,
            "replay_warmup": 100,
            "replay_push_once": False,
            "replay": False,
            "seed": 0,
            "use_data_loader": True,
            "validate_every": 0,
            "wandb": False,
            "random_action_period_init": 0,
            "random_action_period_final": 0,
            "random_action_prob": 0,
            "random_action_init": 0,
            "replay_thread": True,
            "sampling_timeout": 0,
        }

    def setup(self):
        raise NotImplementedError()

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def _wrap_model_mp(self, model, send_to_device=False, thread=True):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers."""
        if send_to_device:
            model.to(self.device)
        if self.num_workers > 0:
            placeholder, keepalive = mp_object_wrapper(
                model,
                self.num_workers,
                cast_types=(gd.Batch, GraphActionCategorical),
                pickle_messages=self.pickle_messages,
                thread=thread,
            )
            self.keepalive.append(keepalive)
            return placeholder, torch.device("cpu")
        return model, self.device

    def build_callbacks(self):
        return {}

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.sampling_model)
        if self.replay_buffer is not None:
            replay_buffer, _ = self._wrap_model_mp(
                self.replay_buffer,
                send_to_device=False,
                thread=self.hps["replay_thread"],
            )
        else:
            replay_buffer = None

        iterator = SamplingIterator(
            self.training_data,
            model,
            self.mb_size,
            self.ctx,
            self.algo,
            self.task,
            dev,
            rp_push_one=self.hps["replay_push_once"],
            replay_buffer=replay_buffer,
            ratio=self.offline_ratio,
            log_dir=os.path.join(self.hps["log_dir"], "train"),
            random_action_prob=self.rp,
        )

        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        if not self.hps["use_data_loader"] or (self.num_workers == 0):
            return iterator
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            timeout=self.hps["sampling_timeout"]
            # The 2 here is an odd quirk of torch 1.10, it is fixed and
            # replaced by None in torch 2.
            # prefetch_factor=1 if self.num_workers else 2,
        )

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.model)
        iterator = SamplingIterator(
            self.test_data,
            model,
            self.mb_size,
            self.ctx,
            self.algo,
            self.task,
            dev,
            replay_buffer=None,
            rp_push_one=self.hps["replay_push_once"],
            ratio=self.valid_offline_ratio,
            log_dir=os.path.join(self.hps["log_dir"], "valid"),
            sample_cond_info=self.hps.get("valid_sample_cond_info", True),
            stream=False,
            random_action_prob=Value(
                "d", self.hps.get("valid_random_action_prob", 0.0)
            ),
        )
        for hook in self.valid_sampling_hooks:
            iterator.add_log_hook(hook)
        if not self.hps["use_data_loader"] or (self.num_workers == 0):
            return iterator
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            timeout=self.hps["sampling_timeout"]
            # prefetch_factor=1 if self.num_workers else 2,
        )

    def train_batch(
        self, batch: gd.Batch, epoch_idx: int, batch_idx: int
    ) -> Dict[str, Any]:
        start_time = time.time()
        self.model.train()
        try:
            loss, info = self.algo.compute_batch_losses(
                self.model, self.target_model, batch
            )
            if not torch.isfinite(loss):
                raise ValueError("loss is not finite")
            if self._validate_parameters and not all(
                [torch.isfinite(i).all() for i in self.model.parameters()]
            ):
                raise ValueError("parameters are not finite")
            step_info = self.step(loss)
        except ValueError as e:
            os.makedirs(self.hps["log_dir"], exist_ok=True)
            torch.save(
                [self.model.state_dict(), batch, loss, info],
                open(self.hps["log_dir"] + "/dump.pkl", "wb"),
            )
            raise e

        if step_info is not None:
            info.update(step_info)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        info["train_batch_time"] = time.time() - start_time
        return info

    def evaluate_batch(
        self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0
    ) -> Dict[str, Any]:
        self.model.eval()
        loss, info = self.algo.compute_batch_losses(
            self.model, self.target_model, batch
        )
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return info

    def set_rp(self, it):
        irp = self.hps["random_action_period_init"]
        ir = self.hps["random_action_init"]

        frp = self.hps["random_action_period_final"]
        fr = self.hps["random_action_prob"]
        if it < irp:
            self.rp.value = ir
        elif it < frp:
            self.rp.value = (frp - it) / (frp - irp) * (ir - fr) + fr
        else:
            self.rp.value = fr

    def diag(self):
        info = {}
        info["lr"] = self.lr_sched.get_last_lr()[0]
        info["mem_allocated"] = torch.cuda.memory_allocated() / byte_to_gb
        info["mem_reserved"] = torch.cuda.memory_reserved() / byte_to_gb
        info["mem_peak"] = torch.cuda.max_memory_allocated() / byte_to_gb
        return info

    def run(self, logger=None):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        if logger is None:
            logger = create_logger(logfile=self.hps["log_dir"] + "/train.log")

        self.target_model: nn.Module
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        valid_freq = self.hps["validate_every"]
        # If checkpoint_every is not specified, checkpoint at every validation epoch
        ckpt_freq = self.hps.get("checkpoint_every", valid_freq)
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        callbacks = self.build_callbacks()
        start = self.hps.get("start_at_step", 0) + 1
        logger.info("Starting training")
        max_it = self.hps["num_training_steps"]
        if self.hps["max_samples"] not in [0, float("inf")]:
            max_it = min(
                max_it,
                ceil(
                    self.hps["max_samples"] / (self.mb_size * (1 - self.offline_ratio))
                ),
            )

        start_time = time.time()
        warmed_up = False
        for it, batch in zip(range(start, 1 + max_it), cycle(train_dl)):
            self.set_rp(it)

            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if (
                self.target_model is not self.model
                and self.hps["update_target_every"] != 0
                and it % self.hps["update_target_every"] == 0
            ):
                self.target_model.load_state_dict(self.model.state_dict())

            if self.hps["reset"] > 0 and it % self.hps["reset"] == 0:
                self.setup_model()
                self.model.to(self.device)
                self.setup_optim()

            if (
                not warmed_up
                and self.replay_buffer is not None
                and self.replay_buffer.shared_cap.value < self.replay_buffer.warmup
            ):
                logger.info(
                    "warming up replay buffer %d/%d",
                    self.replay_buffer.shared_cap.value,
                    self.replay_buffer.warmup,
                )
                continue
            else:
                warmed_up = True
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx)
            info.update(self.diag())
            info["train_time"] = time.time() - start_time
            if self.replay_buffer is not None:
                info["replay_buffer_size"] = self.replay_buffer.shared_cap.value
            start_time = time.time()
            self.log(info, it, "train")
            if self.verbose:
                logger.info(
                    f"iteration {it} : "
                    + " ".join(f"{k}:{v:.2f}" for k, v in info.items())
                )
            if valid_freq > 0 and it % valid_freq == 0:
                acc = defaultdict(list)
                for batch in valid_dl:
                    info = self.evaluate_batch(
                        batch.to(self.device), epoch_idx, batch_idx
                    )
                    for k, v in info.items():
                        acc[k].append(v)
                    logger.info(
                        f"validation - iteration {it} : "
                        + " ".join(f"{k}:{v:.2f}" for k, v in info.items())
                    )
                self.log(
                    {
                        k: np.mean([v.item() if hasattr(v, "item") else v for v in vs])
                        for k, vs in acc.items()
                    },
                    it,
                    "validation",
                )
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, "valid_end")
            elif valid_freq > 0 and it > self.hps["validate_every"]:
                # for wandb
                self.log(end_metrics, it, "valid_end2")

            if ckpt_freq > 0 and it % ckpt_freq == 0:
                self._save_state(it)

        self._save_state(self.hps["num_training_steps"])

    def _save_state(self, it):
        torch.save(
            {
                "models_state_dict": [self.model.state_dict()],
                "hps": self.hps,
                "step": it,
            },
            open(pathlib.Path(self.hps["log_dir"]) / "model_state.pt", "wb"),
        )

    def log(self, info: dict, index: int, key: str):
        info = {k: v.item() if type(v) == torch.Tensor else v for k, v in info.items()}
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(
                self.hps["log_dir"]
            )
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{key}_{k}", v, index)
        if self.wandb:
            import wandb

            wandb.log(dict([(f"{key}/{k}", v) for k, v in info.items()]), step=index)

    def end(self):
        while len(self.keepalive) > 0:
            x = self.keepalive.pop()
            x.set()


def cycle(it):
    while True:
        for i in it:
            yield i
