from typing import Any, Dict, Optional

import numpy as np
from gflownet.algo.soft_q_learning import SoftQLearning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.envs.graph_building_env import GraphActionCategorical, generate_forward_trajectory
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.graph_building_env import GraphBuildingEnvContext


def gumbel_loss(pred, label, beta, clip):
    def f(z):
        torch.exp(z) - z - 1

    def df(z):
        return torch.exp(z) - 1

    def d2f(z):
        return torch.exp(z)

    z = (pred - label) / beta

    loss = f(z)
    loss = torch.where(z <= clip, loss, f(clip) + df(clip) * (z - clip) + 0.5 * d2f(clip) * (z - clip) ** 2)
    loss = torch.where(z >= -clip, loss, f(-clip) + df(-clip) * (z + clip) + 0.5 * d2f(-clip) * (z + clip) ** 2)
    return loss


def asym_loss(pred, label):
    z = pred - label
    zz = z * z
    return (z > 0) * 2 * zz + (z <= 0) * zz


class TrajQLearning(SoftQLearning):
    def compute_batch_losses(
        self, model: nn.Module, target_model: Optional[nn.Module], batch: gd.Batch, num_bootstrap: int = 0
    ):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device

        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        if self.exp:
            rewards = torch.exp(batch.log_rewards)
        else:
            rewards = batch.log_rewards
        rewards = rewards / self.alpha

        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and per molecule predictions
        # Here we will interpret the logits of the fwd_cat as Q values
        Q: GraphActionCategorical
        Q, _ = model(batch, cond_info[batch_idx])
        V0 = model.logZ(cond_info)

        # Here were are again hijacking the GraphActionCategorical machinery to get Q[s,a], but
        # instead of logprobs we're just going to use the logits, i.e. the Q values.
        lp = Q.logsoftmax()
        ts, t = Q.get_temp()

        lp = [x * y for (x, y) in zip(lp, ts)]

        lp = Q.log_prob(batch.actions, logprobs=lp)

        lp_traj = scatter(lp, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        losses = V0 + lp_traj - rewards
        losses = losses * losses

        loss = losses.mean()

        invalid_mask = 1 - batch.is_valid

        info = {
            "mean_loss": loss,
            "offline_loss": losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            "online_loss": losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "invalid_losses": (invalid_mask * losses).sum() / (invalid_mask.sum() + 1e-4),
            "mean_target": V0.mean(),
            "mean_ret": torch.mean(rewards),
            "entropy": Q.entropy().sum(),
            "batch_len": torch.mean(batch.traj_lens.float()),
        }

        return loss, info

    def calc_targets(self, batch, rewards, final_graph_idx, Q: GraphActionCategorical, V_soft=None):
        if V_soft is None:
            V_soft = Q.logsumexp()
        # We now need to compute the target, \hat Q = R_t + V_soft(s_t+1)
        # Shift t+1-> t, pad last state with a 0, multiply by gamma
        shifted_V_soft = self.gamma * torch.cat([V_soft[1:], torch.zeros_like(V_soft[:1])])
        # Replace V(s_T) with R(tau). Since we've shifted the values in the array, V(s_T) is V(s_0)
        # of the next trajectory in the array, and rewards are terminal (0 except at s_T).
        shifted_V_soft[final_graph_idx] = rewards + (1 - batch.is_valid) * self.invalid_penalty
        return shifted_V_soft

    def tree_backup(self, batch, rewards, final_graph_idx, Q: GraphActionCategorical):
        first_v = self.calc_targets(batch, rewards, final_graph_idx, Q)
        shifted_V_soft = first_v.clone()

        cache = True
        Q.logprobs = None
        for _ in range(min(batch.traj_lens.max().item(), self.n_step)):
            # create new vals and a mask for the new vals
            Q.logits, cache = Q.inv_log_prob(batch.actions, shifted_V_soft, cache=cache, original=Q.logits)
            shifted_V_soft = self.calc_targets(batch, rewards, final_graph_idx, Q)

        return first_v, shifted_V_soft
