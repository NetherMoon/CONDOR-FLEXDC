"""CONDOR Set-Transformer adapted to FlexDC behavior labels (v2).

This is the user's CONDOR-based model, not Claude's model.  It keeps the
CONDOR set-attention/PMA/residual-trunk idea while improving the feature
representation and training-facing output heads.

Direct neural outputs
---------------------
* log(mean normalized tracking error + floor)
* log(p90 normalized tracking error + floor)
* one QoS violation probability P_j per real job type

All monetary costs and the paper objective are reconstructed analytically by
``am_flexdc_behavior_training_utilities_v2.py``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FlexDCBehaviorModelConfig:
    """Configuration for the CONDOR-based behavior surrogate.

    ``dim_job_mix=13`` and ``dim_dc_features=12`` correspond to the engineered
    FlexDC features produced by the v2 utilities.  The workload encoder remains
    a mask-aware Set Transformer and therefore accepts variable J.
    """

    dim_job_mix: int = 13
    dim_dc_features: int = 12
    st_dim_hidden: int = 512
    st_num_heads: int = 4
    st_num_outputs: int = 1
    global_projection_dim: int = 128
    linear_dim_hidden: int = 512
    qos_projection_dim: int = 256
    skip_connections: bool = True
    layer_norm: bool = True
    include_masked_mean_pool: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class MaskedMAB(nn.Module):
    """Mask-aware Multihead Attention Block used by CONDOR/Set Transformer.

    Masks use ``True`` for real job tokens and ``False`` for padded slots.
    Padded keys cannot receive attention and padded queries are zeroed after
    every block.
    """

    def __init__(
        self,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int,
        *,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if dim_v % num_heads != 0:
            raise ValueError(f"dim_v={dim_v} must be divisible by num_heads={num_heads}")
        self.dim_v = int(dim_v)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim_v // self.num_heads
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)
        self.ln0 = nn.LayerNorm(dim_v) if layer_norm else None
        self.ln1 = nn.LayerNorm(dim_v) if layer_norm else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, head_dim = x.shape
        return x.transpose(1, 2).contiguous().reshape(batch, length, heads * head_dim)

    def forward(
        self,
        q_input: torch.Tensor,
        k_input: torch.Tensor,
        *,
        key_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.fc_q(q_input))
        k = self._split_heads(self.fc_k(k_input))
        v = self._split_heads(self.fc_v(k_input))

        # Standard scaled dot-product attention.  Using the per-head dimension
        # avoids attention becoming unnecessarily flat at large hidden widths.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_mask is not None:
            key_mask = key_mask.to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(
                ~key_mask[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )

        attention = torch.softmax(scores, dim=-1)
        output = q + torch.matmul(attention, v)
        output = self._merge_heads(output)
        if self.ln0 is not None:
            output = self.ln0(output)
        output = output + F.silu(self.fc_o(output))
        if self.ln1 is not None:
            output = self.ln1(output)

        if query_mask is not None:
            output = output * query_mask.to(output.dtype).unsqueeze(-1)
        return output


class MaskedSAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, *, layer_norm: bool = True) -> None:
        super().__init__()
        self.mab = MaskedMAB(
            dim_in,
            dim_in,
            dim_out,
            num_heads,
            layer_norm=layer_norm,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mab(x, x, key_mask=mask, query_mask=mask)


class MaskedPMA(nn.Module):
    """Pooling by Multihead Attention with correct padding masks."""

    def __init__(self, dim: int, num_heads: int, num_seeds: int, *, layer_norm: bool = True) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed)
        self.mab = MaskedMAB(
            dim,
            dim,
            dim,
            num_heads,
            layer_norm=layer_norm,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seeds = self.seed.expand(x.size(0), -1, -1)
        query_mask = torch.ones(
            (x.size(0), seeds.size(1)),
            dtype=torch.bool,
            device=x.device,
        )
        return self.mab(seeds, x, key_mask=mask, query_mask=query_mask)


class DataCenterBehaviorModel(nn.Module):
    """CONDOR Set Transformer with FlexDC behavior output heads."""

    def __init__(self, config: FlexDCBehaviorModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or FlexDCBehaviorModelConfig()
        c = self.config
        self.skip_connections = bool(c.skip_connections)
        self.include_masked_mean_pool = bool(c.include_masked_mean_pool)
        self.activation = nn.SiLU()

        self.sab1 = MaskedSAB(
            c.dim_job_mix,
            c.st_dim_hidden,
            c.st_num_heads,
            layer_norm=c.layer_norm,
        )
        self.sab2 = MaskedSAB(
            c.st_dim_hidden,
            c.st_dim_hidden,
            c.st_num_heads,
            layer_norm=c.layer_norm,
        )
        self.pma = MaskedPMA(
            c.st_dim_hidden,
            c.st_num_heads,
            c.st_num_outputs,
            layer_norm=c.layer_norm,
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(c.dim_dc_features, c.global_projection_dim),
            nn.SiLU(),
            nn.Linear(c.global_projection_dim, c.global_projection_dim),
            nn.SiLU(),
        )

        pooled_dim = c.st_dim_hidden
        if self.include_masked_mean_pool:
            pooled_dim += c.st_dim_hidden
        fused_dim = pooled_dim + c.global_projection_dim

        self.linear1 = nn.Linear(fused_dim, c.linear_dim_hidden)
        self.linear2 = nn.Linear(c.linear_dim_hidden, c.linear_dim_hidden)
        self.linear3 = nn.Linear(c.linear_dim_hidden, c.linear_dim_hidden)
        self.tracking_output = nn.Linear(c.linear_dim_hidden, 2)

        # Shared per-job head.  Each P_j sees its own encoded token plus the
        # pooled bid/workload context, so the head can represent competition
        # between queues while still emitting one probability per real job.
        self.qos_token_projection = nn.Linear(c.st_dim_hidden, c.qos_projection_dim)
        self.qos_context_projection = nn.Linear(c.linear_dim_hidden, c.qos_projection_dim)
        self.qos_head = nn.Sequential(
            nn.Linear(2 * c.qos_projection_dim, c.qos_projection_dim),
            nn.SiLU(),
            nn.Linear(c.qos_projection_dim, 1),
        )

    @staticmethod
    def infer_mask(workload_mix: torch.Tensor) -> torch.Tensor:
        if workload_mix.ndim != 3:
            raise ValueError(
                f"workload_mix must have shape [B,J,F], got {tuple(workload_mix.shape)}"
            )
        return workload_mix.abs().sum(dim=-1) > 0

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(x.dtype).unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        sim_features: torch.Tensor,
        workload_mix: torch.Tensor,
        workload_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if sim_features.ndim != 2:
            raise ValueError(f"sim_features must have shape [B,F], got {tuple(sim_features.shape)}")
        if sim_features.size(-1) != self.config.dim_dc_features:
            raise ValueError(
                f"Expected {self.config.dim_dc_features} global features, got {sim_features.size(-1)}"
            )
        if workload_mix.size(-1) != self.config.dim_job_mix:
            raise ValueError(
                f"Expected {self.config.dim_job_mix} token features, got {workload_mix.size(-1)}"
            )

        if workload_mask is None:
            workload_mask = self.infer_mask(workload_mix)
        workload_mask = workload_mask.to(dtype=torch.bool, device=workload_mix.device)
        if torch.any(workload_mask.sum(dim=1) == 0):
            raise ValueError("Every sample must contain at least one real job token.")

        token_features = self.sab1(workload_mix, workload_mask)
        token_features = self.sab2(token_features, workload_mask)

        pma_pool = self.pma(token_features, workload_mask).mean(dim=1)
        pools = [pma_pool]
        if self.include_masked_mean_pool:
            pools.append(self.masked_mean(token_features, workload_mask))
        global_context = self.global_encoder(sim_features)
        x1 = torch.cat([*pools, global_context], dim=-1)

        x2 = self.activation(self.linear1(x1))
        x3 = self.activation(self.linear2(x2))
        if self.skip_connections:
            x4 = self.activation(self.linear3(x3 + x2))
            context = x4 + x3
        else:
            context = self.activation(self.linear3(x3))

        tracking_logs = self.tracking_output(context)

        token_qos = self.qos_token_projection(token_features)
        context_qos = self.qos_context_projection(context).unsqueeze(1).expand(
            -1,
            token_features.size(1),
            -1,
        )
        qos_logits = self.qos_head(torch.cat([token_qos, context_qos], dim=-1)).squeeze(-1)
        qos_probabilities = torch.sigmoid(qos_logits)
        qos_probabilities = qos_probabilities * workload_mask.to(qos_probabilities.dtype)

        return {
            "tracking_logs": tracking_logs,
            "qos_logits": qos_logits,
            "qos_probabilities": qos_probabilities,
            "workload_mask": workload_mask,
            "token_features": token_features,
            "context_features": context,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
