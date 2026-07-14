"""CONDOR Set-Transformer surrogate adapted to FlexDC raw behavior labels.

This module intentionally preserves CONDOR's high-level architecture:

    variable-length workload set -> self-attention -> attention pooling
    -> residual MLP fused with P/R/data-center features

The output contract is changed for FlexDC:

    * one scalar log mean normalized tracking error
    * one scalar log p90 normalized tracking error
    * one QoS violation probability per real job type

Costs are deliberately not direct neural-network outputs. They are reconstructed
from these raw behavior predictions using the known FlexDC equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FlexDCBehaviorModelConfig:
    dim_job_mix: int = 7
    dim_dc_features: int = 5
    st_dim_hidden: int = 512
    st_dim_output: int = 1019  # preserves CONDOR's 1024-wide fused representation
    st_num_heads: int = 4
    st_num_outputs: int = 1
    linear_dim_hidden: int = 512
    qos_projection_dim: int = 256
    skip_connections: bool = True
    layer_norm: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class MaskedMAB(nn.Module):
    """Mask-aware version of CONDOR/Set-Transformer MAB.

    key_mask/query_mask use True for real job tokens and False for padding.
    """

    def __init__(self, dim_q: int, dim_k: int, dim_v: int, num_heads: int, ln: bool = False):
        super().__init__()
        if dim_v % num_heads != 0:
            raise ValueError(f"dim_v={dim_v} must be divisible by num_heads={num_heads}")
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.head_dim = dim_v // num_heads
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)
        self.ln0 = nn.LayerNorm(dim_v) if ln else None
        self.ln1 = nn.LayerNorm(dim_v) if ln else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, heads * head_dim)

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

        # Preserve the scaling used by the released CONDOR Set Transformer code.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_v)
        if key_mask is not None:
            key_mask = key_mask.to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        attention = torch.softmax(scores, dim=-1)
        output = q + torch.matmul(attention, v)
        output = self._merge_heads(output)
        if self.ln0 is not None:
            output = self.ln0(output)
        output = output + F.relu(self.fc_o(output))
        if self.ln1 is not None:
            output = self.ln1(output)

        if query_mask is not None:
            output = output * query_mask.to(output.dtype).unsqueeze(-1)
        return output


class MaskedSAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False):
        super().__init__()
        self.mab = MaskedMAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mab(x, x, key_mask=mask, query_mask=mask)


class MaskedPMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int, ln: bool = False):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed)
        self.mab = MaskedMAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seeds = self.seed.expand(x.size(0), -1, -1)
        query_mask = torch.ones(
            (x.size(0), seeds.size(1)), dtype=torch.bool, device=x.device
        )
        return self.mab(seeds, x, key_mask=mask, query_mask=query_mask)


class DataCenterBehaviorModel(nn.Module):
    """CONDOR Set Transformer with FlexDC behavior output heads."""

    def __init__(self, config: FlexDCBehaviorModelConfig | None = None):
        super().__init__()
        self.config = config or FlexDCBehaviorModelConfig()
        c = self.config
        self.skip_connections = c.skip_connections
        self.activation = nn.Softplus()

        self.sab1 = MaskedSAB(c.dim_job_mix, c.st_dim_hidden, c.st_num_heads, ln=c.layer_norm)
        self.sab2 = MaskedSAB(c.st_dim_hidden, c.st_dim_hidden, c.st_num_heads, ln=c.layer_norm)
        self.pma = MaskedPMA(c.st_dim_hidden, c.st_num_heads, c.st_num_outputs, ln=c.layer_norm)
        self.pool_projection = nn.Linear(c.st_dim_hidden, c.st_dim_output)

        fused_dim = c.st_dim_output + c.dim_dc_features
        self.linear1 = nn.Linear(fused_dim, c.linear_dim_hidden)
        self.linear2 = nn.Linear(c.linear_dim_hidden, c.linear_dim_hidden)
        self.linear3 = nn.Linear(c.linear_dim_hidden, c.linear_dim_hidden)

        # Two workload-level scalar outputs: log mean tracking and log p90 tracking.
        self.tracking_output = nn.Linear(c.linear_dim_hidden, 2)

        # A shared per-job QoS head. The same function is applied to every real token.
        self.qos_token_projection = nn.Linear(c.st_dim_hidden, c.qos_projection_dim)
        self.qos_context_projection = nn.Linear(c.linear_dim_hidden, c.qos_projection_dim)
        self.qos_head = nn.Sequential(
            nn.Linear(2 * c.qos_projection_dim, c.qos_projection_dim),
            nn.Softplus(),
            nn.Linear(c.qos_projection_dim, 1),
        )

    @staticmethod
    def infer_mask(workload_mix: torch.Tensor) -> torch.Tensor:
        if workload_mix.ndim != 3:
            raise ValueError(f"workload_mix must have shape [B,J,F], got {workload_mix.shape}")
        return workload_mix.abs().sum(dim=-1) > 0

    def forward(
        self,
        sim_features: torch.Tensor,
        workload_mix: torch.Tensor,
        workload_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if workload_mask is None:
            workload_mask = self.infer_mask(workload_mix)
        workload_mask = workload_mask.to(dtype=torch.bool, device=workload_mix.device)

        token_features = self.sab1(workload_mix, workload_mask)
        token_features = self.sab2(token_features, workload_mask)

        pooled = self.pma(token_features, workload_mask)
        pooled = self.pool_projection(pooled).squeeze(1)

        x1 = torch.cat((pooled, sim_features), dim=-1)
        x2 = self.activation(self.linear1(x1))
        x3 = self.activation(self.linear2(x2))
        if self.skip_connections:
            x4 = self.activation(self.linear3(x3 + x2))
            context = x4 + x3
        else:
            x4 = self.activation(self.linear3(x3))
            context = x4

        tracking_logs = self.tracking_output(context)

        token_qos = self.qos_token_projection(token_features)
        context_qos = self.qos_context_projection(context).unsqueeze(1).expand(-1, token_features.size(1), -1)
        qos_logits = self.qos_head(torch.cat((token_qos, context_qos), dim=-1)).squeeze(-1)
        qos_probabilities = torch.sigmoid(qos_logits)
        qos_probabilities = qos_probabilities * workload_mask.to(qos_probabilities.dtype)

        return {
            "tracking_logs": tracking_logs,
            "qos_logits": qos_logits,
            "qos_probabilities": qos_probabilities,
            "workload_mask": workload_mask,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
