"""Generic inference and optimization utilities for the CONDOR-FlexDC behavior model.

This module is intentionally paired with:

* ``data_center_model_flexdc_behavior.py``
* ``flexdc_behavior_training_utilities.py``

It reuses the training feature definitions and checkpoint metadata so that
prediction and gradient-based optimization use exactly the same 13 per-job
features, 12 global features, standardization statistics, label decoding, and
FlexDC objective constants as model training.

The neural model directly predicts:

* log mean normalized tracking error;
* log p90 normalized tracking error;
* one QoS violation probability P_j per real job type.

Known FlexDC monetary and penalty terms are reconstructed analytically.
"""

from __future__ import annotations

import configparser
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from flexdc_behavior_training_utilities import (
    BehaviorDataMetadata,
    FlexDCBehaviorConstants,
    build_global_features_raw,
    build_token_features_raw,
    decode_tracking_logs,
    load_behavior_model_checkpoint,
    parse_list,
)
from data_center_model_flexdc_behavior import (
    DataCenterBehaviorModel,
    FlexDCBehaviorModelConfig,
)

KWH_IN_WATT_SECONDS = 3600.0 * 1000.0
DEFAULT_ENERGY_PRICE_PER_KWH = 0.1
DEFAULT_WEIGHT_MIN_FRACTION_OF_EQUAL = 0.1
DEFAULT_WEIGHT_MAX_MULTIPLE_OF_EQUAL = 4.0


@dataclass(frozen=True)
class WorkloadSpec:
    path: str
    job_names: list[str]
    mix: np.ndarray

    @property
    def job_count(self) -> int:
        return int(self.mix.shape[0])

    @property
    def pmin_kw_per_server(self) -> float:
        return float(np.min(self.mix[:, 0]) / 1000.0)

    @property
    def pmax_kw_per_server(self) -> float:
        return float(np.max(self.mix[:, 1]) / 1000.0)


@dataclass(frozen=True)
class ExperimentSpec:
    path: str
    server_count: int
    utilization: float
    idle_watts: float
    simulation_duration_seconds: int
    random_seed: int
    iso_file_path: str
    iso_signal_start_hour: int | None


@dataclass(frozen=True)
class PRBounds:
    pmin_kw_per_server: float
    pmax_kw_per_server: float
    pbar_lower_kw_per_server: float
    pbar_upper_kw_per_server: float
    pr_upper_kw_per_server: float
    r_lower_kw_per_server: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WeightBounds:
    equal_weight: float
    relative_lower: float
    relative_upper: float
    server_lower: float
    upper_from_lower: float
    final_lower: float
    final_upper: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SafetyLimits:
    exact_tracking_threshold: float
    exact_qos_threshold: float
    selection_tracking_limit: float
    selection_qos_limit: float

    @property
    def tracking_margin(self) -> float:
        return float(self.exact_tracking_threshold - self.selection_tracking_limit)

    @property
    def qos_margin(self) -> float:
        return float(self.exact_qos_threshold - self.selection_qos_limit)

    def to_dict(self) -> dict:
        return asdict(self) | {
            "tracking_margin": self.tracking_margin,
            "qos_margin": self.qos_margin,
        }


@dataclass(frozen=True)
class LoadedBehaviorModel:
    model: DataCenterBehaviorModel
    checkpoint: dict
    metadata: BehaviorDataMetadata
    constants: FlexDCBehaviorConstants
    device: torch.device
    checkpoint_path: str


@dataclass
class OptimizationSettings:
    starts: int = 512
    iterations: int = 1500
    learning_rate: float = 0.03
    minimum_learning_rate: float = 5e-4
    mode: str = "margin_constrained"  # pure_objective, exact_constrained, margin_constrained
    tracking_penalty: float = 2000.0
    qos_penalty: float = 2000.0
    penalty_ramp_fraction: float = 0.30
    top_k: int = 5
    candidate_distance: float = 0.03
    random_seed: int = 0
    near_equal_start_fraction: float = 0.25
    high_p_low_r_start_fraction: float = 0.25
    enforce_flexdc_weight_bounds: bool = True
    weight_min_fraction_of_equal: float = DEFAULT_WEIGHT_MIN_FRACTION_OF_EQUAL
    weight_max_multiple_of_equal: float = DEFAULT_WEIGHT_MAX_MULTIPLE_OF_EQUAL
    # Optional stricter experiment-level overrides.  These are intersected
    # with the automatic FlexDC bounds rather than replacing them.
    weight_min: float | None = None
    weight_max: float | None = None
    r_over_p_max: float | None = None
    pbar_min_override: float | None = None
    pbar_max_override: float | None = None
    r_min_override: float | None = None
    r_max_override: float | None = None
    # Optional exact ratio starts used by the original paired-comparison
    # notebook. Each tuple is (Pbar_ratio, R_ratio), converted with the same
    # workload/experiment denominators used by model features. Remaining starts
    # retain the normal deterministic random/near-equal initialization.
    explicit_ratio_starts: tuple[tuple[float, float], ...] | None = None
    illegal_start_policy: str = "project"  # project, skip, strict
    log_every: int = 25

    def validate(self, job_count: int) -> None:
        if self.starts < 1:
            raise ValueError("starts must be at least 1")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.learning_rate <= 0 or self.minimum_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.mode not in {"pure_objective", "exact_constrained", "margin_constrained"}:
            raise ValueError(f"Unsupported optimization mode: {self.mode}")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 <= self.near_equal_start_fraction <= 1:
            raise ValueError("near_equal_start_fraction must be in [0,1]")
        if not 0 <= self.high_p_low_r_start_fraction <= 1:
            raise ValueError("high_p_low_r_start_fraction must be in [0,1]")
        if self.weight_min_fraction_of_equal < 0:
            raise ValueError("weight_min_fraction_of_equal must be nonnegative")
        if self.weight_max_multiple_of_equal <= 0:
            raise ValueError("weight_max_multiple_of_equal must be positive")
        if self.weight_min is not None:
            if self.weight_min < 0 or self.weight_min * job_count >= 1.0:
                raise ValueError("weight_min must satisfy 0 <= J*weight_min < 1")
        if self.weight_max is not None:
            if not 0 < self.weight_max <= 1:
                raise ValueError("weight_max must be in (0,1]")
            if self.weight_min is not None and self.weight_max <= self.weight_min:
                raise ValueError("weight_max must exceed weight_min")
        if self.r_over_p_max is not None and not 0 < self.r_over_p_max <= 1:
            raise ValueError("r_over_p_max must be in (0,1]")
        if self.illegal_start_policy not in {"project", "skip", "strict"}:
            raise ValueError("illegal_start_policy must be project, skip, or strict")
        if self.explicit_ratio_starts is not None:
            if len(self.explicit_ratio_starts) > self.starts:
                raise ValueError("starts must be >= the number of explicit_ratio_starts")
            for pair in self.explicit_ratio_starts:
                if len(pair) != 2 or not all(math.isfinite(float(value)) for value in pair):
                    raise ValueError(f"Invalid explicit ratio start: {pair}")


# ---------------------------------------------------------------------------
# Configuration parsing and physical bounds
# ---------------------------------------------------------------------------


def read_workload_config(path: str | Path) -> WorkloadSpec:
    path = Path(path).resolve()
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(path)
    default_job_size = parser.defaults().get("job_size", "1")
    rows: list[list[float]] = []
    names: list[str] = []
    for section in parser.sections():
        job = parser[section]
        rows.append(
            [
                job.getfloat("min_job_power_watts"),
                job.getfloat("max_job_power_watts"),
                job.getfloat("min_time_seconds"),
                job.getfloat("max_time_seconds"),
                job.getfloat("qos_constraint"),
                float(job.get("job_size", default_job_size)),
            ]
        )
        names.append(section)
    if not rows:
        raise ValueError(f"No workload job sections found in {path}")
    return WorkloadSpec(path=str(path), job_names=names, mix=np.asarray(rows, dtype=np.float64))


def read_experiment_config(
    path: str | Path,
    *,
    server_count_override: int | None = None,
    utilization_override: float | None = None,
) -> ExperimentSpec:
    path = Path(path).resolve()
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(path)
    if "system" not in parser or "iso" not in parser:
        raise ValueError(f"Expected [system] and [iso] sections in {path}")
    system = parser["system"]
    iso = parser["iso"]
    server_count = int(server_count_override if server_count_override is not None else system.getint("server_count"))
    utilization = float(utilization_override if utilization_override is not None else system.getfloat("utilization"))
    if server_count <= 0:
        raise ValueError("server_count must be positive")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must lie in (0,1]")
    return ExperimentSpec(
        path=str(path),
        server_count=server_count,
        utilization=utilization,
        idle_watts=system.getfloat("idle_watts"),
        simulation_duration_seconds=system.getint("simulation_duration"),
        random_seed=system.getint("random_seed"),
        iso_file_path=iso.get("iso_file_path"),
        iso_signal_start_hour=iso.getint("iso_signal_start_hour", fallback=None),
    )


def calculate_pr_bounds(
    workload: WorkloadSpec,
    *,
    pbar_lower_factor: float = 0.9,
    pbar_upper_factor: float = 1.0,
    pr_upper_factor: float = 1.2,
    r_lower_kw_per_server: float = 0.01,
) -> PRBounds:
    pmin = workload.pmin_kw_per_server
    pmax = workload.pmax_kw_per_server
    lower = float(pbar_lower_factor) * pmin
    upper = float(pbar_upper_factor) * pmax
    pr_upper = float(pr_upper_factor) * pmax
    r_lower = float(r_lower_kw_per_server)
    upper = min(upper, pr_upper - r_lower)
    if lower <= 0 or upper <= lower:
        raise ValueError(f"Invalid Pbar interval [{lower}, {upper}]")
    if r_lower <= 0 or pr_upper <= lower + r_lower:
        raise ValueError("Invalid R/combined bid bounds")
    return PRBounds(
        pmin_kw_per_server=pmin,
        pmax_kw_per_server=pmax,
        pbar_lower_kw_per_server=lower,
        pbar_upper_kw_per_server=upper,
        pr_upper_kw_per_server=pr_upper,
        r_lower_kw_per_server=r_lower,
    )


def calculate_pr_denominators(
    workload: WorkloadSpec,
    experiment: ExperimentSpec,
) -> tuple[float, float]:
    avg_max_job_power = float(np.mean(workload.mix[:, 1]))
    active_servers = float(experiment.server_count) * float(experiment.utilization)
    idle_servers = float(experiment.server_count) - active_servers
    pbar_denominator = avg_max_job_power * active_servers + experiment.idle_watts * idle_servers
    r_denominator = (
        avg_max_job_power * experiment.server_count
        - experiment.idle_watts * experiment.server_count
    ) / 2.0
    if pbar_denominator <= 0 or r_denominator <= 0:
        raise ValueError("Invalid P/R denominators; check workload and idle power")
    return float(pbar_denominator), float(r_denominator)


def calculate_pr_features(
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    workload: WorkloadSpec,
    experiment: ExperimentSpec,
) -> dict:
    pbar_denominator, r_denominator = calculate_pr_denominators(workload, experiment)
    p_actual = float(pbar_kw_per_server) * 1000.0 * experiment.server_count
    r_actual = float(r_kw_per_server) * 1000.0 * experiment.server_count
    return {
        "P_actual_watts": p_actual,
        "R_actual_watts": r_actual,
        "Pbar_ratio": p_actual / pbar_denominator,
        "R_ratio": r_actual / r_denominator,
        "Pbar_denominator_watts": pbar_denominator,
        "R_denominator_watts": r_denominator,
    }


def validate_weights(weights: Sequence[float], job_count: int) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or len(array) != job_count:
        raise ValueError(f"Expected {job_count} weights, got shape {array.shape}")
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError("Weights must be finite and non-negative")
    if not np.isclose(float(array.sum()), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to one; got {array.sum()}")
    return array


def calculate_weight_bounds(
    job_count: int,
    server_count: int,
    *,
    min_fraction_of_equal: float = DEFAULT_WEIGHT_MIN_FRACTION_OF_EQUAL,
    max_multiple_of_equal: float = DEFAULT_WEIGHT_MAX_MULTIPLE_OF_EQUAL,
) -> WeightBounds:
    """Mirror FlexDC's finalized workload-weight bounds exactly.

    FlexDC requires every job type to receive at least a fraction of equal
    allocation and at least one server.  The upper bound is the stricter of
    the configured multiple of equal weight and the amount remaining after
    assigning the lower bound to the other J-1 job types.
    """
    job_count = int(job_count)
    server_count = int(server_count)
    if job_count <= 0:
        raise ValueError("job_count must be positive")
    if server_count < job_count:
        raise ValueError(
            f"Weight feasibility is impossible: server_count={server_count} < job_count={job_count}"
        )
    equal = 1.0 / job_count
    relative_lower = float(min_fraction_of_equal) * equal
    relative_upper = float(max_multiple_of_equal) * equal
    server_lower = 1.0 / server_count
    final_lower = max(relative_lower, server_lower)
    upper_from_lower = 1.0 - (job_count - 1) * final_lower
    final_upper = min(relative_upper, upper_from_lower)
    if final_upper < final_lower or job_count * final_lower > 1.0 + 1e-12 or job_count * final_upper < 1.0 - 1e-12:
        raise ValueError(
            "Infeasible finalized weight bounds: "
            f"J={job_count}, N={server_count}, lower={final_lower}, upper={final_upper}"
        )
    return WeightBounds(
        equal_weight=equal,
        relative_lower=relative_lower,
        relative_upper=relative_upper,
        server_lower=server_lower,
        upper_from_lower=upper_from_lower,
        final_lower=final_lower,
        final_upper=final_upper,
    )


def resolve_effective_weight_bounds(
    settings: OptimizationSettings,
    *,
    job_count: int,
    server_count: int,
) -> WeightBounds:
    if settings.enforce_flexdc_weight_bounds:
        base = calculate_weight_bounds(
            job_count,
            server_count,
            min_fraction_of_equal=settings.weight_min_fraction_of_equal,
            max_multiple_of_equal=settings.weight_max_multiple_of_equal,
        )
        lower = base.final_lower
        upper = base.final_upper
    else:
        equal = 1.0 / job_count
        base = WeightBounds(equal, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        lower, upper = 0.0, 1.0
    if settings.weight_min is not None:
        lower = max(lower, float(settings.weight_min))
    if settings.weight_max is not None:
        upper = min(upper, float(settings.weight_max))
    if job_count * lower >= 1.0 - 1e-12:
        raise ValueError(f"Effective weight lower bound {lower} is infeasible for J={job_count}")
    if job_count * upper < 1.0 - 1e-12:
        raise ValueError(f"Effective weight upper bound {upper} is infeasible for J={job_count}")
    if upper <= lower:
        raise ValueError(f"Effective weight bounds are empty: lower={lower}, upper={upper}")
    return WeightBounds(
        equal_weight=base.equal_weight,
        relative_lower=base.relative_lower,
        relative_upper=base.relative_upper,
        server_lower=base.server_lower,
        upper_from_lower=base.upper_from_lower,
        final_lower=lower,
        final_upper=upper,
    )


def validate_weight_bounds(
    weights: Sequence[float],
    bounds: WeightBounds,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Weights contain non-finite values")
    if abs(float(values.sum()) - 1.0) > tolerance:
        raise ValueError(f"Weights must sum to 1; sum={values.sum()}")
    if float(values.min()) < bounds.final_lower - tolerance:
        raise ValueError(
            f"Weight vector violates FlexDC lower bound: min={values.min()}, lower={bounds.final_lower}, weights={values.tolist()}"
        )
    if float(values.max()) > bounds.final_upper + tolerance:
        raise ValueError(
            f"Weight vector violates FlexDC upper bound: max={values.max()}, upper={bounds.final_upper}, weights={values.tolist()}"
        )
    return values


def validate_physical_bid(
    pbar: float,
    reserve: float,
    bounds: PRBounds,
    *,
    r_over_p_max: float | None = None,
) -> None:
    if not bounds.pbar_lower_kw_per_server - 1e-9 <= pbar <= bounds.pbar_upper_kw_per_server + 1e-9:
        raise ValueError(f"Pbar={pbar} outside [{bounds.pbar_lower_kw_per_server}, {bounds.pbar_upper_kw_per_server}]")
    if reserve < bounds.r_lower_kw_per_server - 1e-9:
        raise ValueError(f"R={reserve} below {bounds.r_lower_kw_per_server}")
    if reserve > pbar + 1e-9:
        raise ValueError("R must not exceed Pbar")
    if pbar + reserve > bounds.pr_upper_kw_per_server + 1e-9:
        raise ValueError("Pbar+R exceeds the workload-specific combined-bid limit")
    if r_over_p_max is not None and reserve > r_over_p_max * pbar + 1e-9:
        raise ValueError(f"R/Pbar exceeds configured maximum {r_over_p_max}")


# ---------------------------------------------------------------------------
# Checkpoint loading and feature construction
# ---------------------------------------------------------------------------


def load_behavior_model(
    checkpoint_path: str | Path,
    *,
    device_name: str = "auto",
) -> LoadedBehaviorModel:
    checkpoint_path = Path(checkpoint_path).resolve()
    model, checkpoint = load_behavior_model_checkpoint(
        checkpoint_path,
        model_class=DataCenterBehaviorModel,
        config_class=FlexDCBehaviorModelConfig,
        device_name=device_name,
    )
    # Inference optimizes Pbar, R, and weight inputs—not the trained network.
    # Freeze model parameters to avoid storing/accumulating 3.9M parameter
    # gradients during every candidate-optimization step. Gradients still flow
    # through the frozen network into the differentiable inputs.
    model.eval()
    model.requires_grad_(False)
    metadata = BehaviorDataMetadata(**checkpoint["data_metadata"])
    constants = FlexDCBehaviorConstants(**metadata.constants)
    device = next(model.parameters()).device
    return LoadedBehaviorModel(
        model=model,
        checkpoint=checkpoint,
        metadata=metadata,
        constants=constants,
        device=device,
        checkpoint_path=str(checkpoint_path),
    )


def make_feature_row(
    *,
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    weights: Sequence[float],
    workload: WorkloadSpec,
    experiment: ExperimentSpec,
) -> tuple[pd.Series, np.ndarray, np.ndarray, dict]:
    weights_array = validate_weights(weights, workload.job_count)
    pr = calculate_pr_features(pbar_kw_per_server, r_kw_per_server, workload, experiment)
    row = pd.Series(
        {
            "Pbar_kw_per_server": float(pbar_kw_per_server),
            "R_kw_per_server": float(r_kw_per_server),
            "Pbar_ratio": float(pr["Pbar_ratio"]),
            "R_ratio": float(pr["R_ratio"]),
            "server_count": int(experiment.server_count),
            "utilization": float(experiment.utilization),
            "workload_mix_size": int(workload.job_count),
        }
    )
    tokens = build_token_features_raw(
        workload.mix,
        weights_array,
        utilization=experiment.utilization,
        server_count=experiment.server_count,
    )
    global_features = build_global_features_raw(row)
    return row, tokens, global_features, pr


def standardize_features(
    tokens: np.ndarray,
    global_features: np.ndarray,
    metadata: BehaviorDataMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    token_mean = np.asarray(metadata.token_feature_mean, dtype=np.float64)
    token_std = np.asarray(metadata.token_feature_std, dtype=np.float64)
    global_mean = np.asarray(metadata.global_feature_mean, dtype=np.float64)
    global_std = np.asarray(metadata.global_feature_std, dtype=np.float64)
    if tokens.shape[1] != len(token_mean):
        raise ValueError("Token feature dimension does not match checkpoint metadata")
    if global_features.shape[0] != len(global_mean):
        raise ValueError("Global feature dimension does not match checkpoint metadata")
    return (
        (tokens - token_mean) / token_std,
        (global_features - global_mean) / global_std,
    )


def resolve_safety_limits(
    constants: FlexDCBehaviorConstants,
    *,
    tracking_limit: float | None = None,
    qos_limit: float | None = None,
    tracking_margin: float = 0.04,
    qos_margin: float = 0.01,
) -> SafetyLimits:
    selection_tracking = (
        float(tracking_limit)
        if tracking_limit is not None
        else float(constants.tracking_threshold - tracking_margin)
    )
    selection_qos = (
        float(qos_limit)
        if qos_limit is not None
        else float(constants.qos_threshold - qos_margin)
    )
    if selection_tracking <= 0 or selection_qos <= 0:
        raise ValueError("Safety limits must be positive")
    if selection_tracking > constants.tracking_threshold + 1e-12:
        raise ValueError("Selection tracking limit is looser than the exact FlexDC threshold")
    if selection_qos > constants.qos_threshold + 1e-12:
        raise ValueError("Selection QoS limit is looser than the exact FlexDC threshold")
    return SafetyLimits(
        exact_tracking_threshold=float(constants.tracking_threshold),
        exact_qos_threshold=float(constants.qos_threshold),
        selection_tracking_limit=selection_tracking,
        selection_qos_limit=selection_qos,
    )


# ---------------------------------------------------------------------------
# Analytical FlexDC reconstruction
# ---------------------------------------------------------------------------


def reconstruct_costs_numpy(
    *,
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    mean_tracking: float,
    p90_tracking: float,
    qos_probabilities: Sequence[float],
    experiment: ExperimentSpec,
    constants: FlexDCBehaviorConstants,
    energy_price_per_kwh: float = DEFAULT_ENERGY_PRICE_PER_KWH,
) -> dict:
    duration = float(experiment.simulation_duration_seconds)
    p_actual = float(pbar_kw_per_server) * 1000.0 * experiment.server_count
    r_actual = float(r_kw_per_server) * 1000.0 * experiment.server_count
    coefficient = float(energy_price_per_kwh) / KWH_IN_WATT_SECONDS
    power_cost = coefficient * (p_actual - r_actual) * duration
    m_track = coefficient * r_actual * float(mean_tracking) * duration
    m_rsr = power_cost + m_track
    ctrack = float(constants.ctrack_psi * np.logaddexp(0.0, constants.ctrack_mu * (float(p90_tracking) - constants.tracking_threshold)))
    probs = np.asarray(qos_probabilities, dtype=np.float64)
    cqos = float(constants.qos_beta * np.logaddexp(0.0, constants.qos_rho * (probs - constants.qos_threshold)).sum())
    return {
        "Simulator_Power_Cost": float(power_cost),
        "Predicted_Mtrack_Cost": float(m_track),
        "Predicted_M_RSR": float(m_rsr),
        "Predicted_Ctrack": float(ctrack),
        "Predicted_CQoS": float(cqos),
        "Predicted_Full_Objective": float(m_rsr + ctrack + cqos),
        "P_actual_watts": float(p_actual),
        "R_actual_watts": float(r_actual),
    }


def _prediction_from_tensors(
    *,
    loaded: LoadedBehaviorModel,
    tracking_logs: torch.Tensor,
    qos_probabilities: torch.Tensor,
    pbar: float,
    reserve: float,
    workload: WorkloadSpec,
    experiment: ExperimentSpec,
    safety: SafetyLimits,
    weights: Sequence[float],
) -> tuple[dict, pd.DataFrame]:
    mean_tracking_t, p90_tracking_t = decode_tracking_logs(tracking_logs, loaded.constants)
    mean_tracking = float(mean_tracking_t[0].detach().cpu())
    p90_tracking = float(p90_tracking_t[0].detach().cpu())
    probs = qos_probabilities[0, : workload.job_count].detach().cpu().numpy().astype(float)
    max_pj = float(np.max(probs))
    costs = reconstruct_costs_numpy(
        pbar_kw_per_server=pbar,
        r_kw_per_server=reserve,
        mean_tracking=mean_tracking,
        p90_tracking=p90_tracking,
        qos_probabilities=probs,
        experiment=experiment,
        constants=loaded.constants,
    )
    exact_tracking_pass = p90_tracking <= safety.exact_tracking_threshold
    exact_qos_pass = max_pj <= safety.exact_qos_threshold
    safety_tracking_pass = p90_tracking <= safety.selection_tracking_limit
    safety_qos_pass = max_pj <= safety.selection_qos_limit
    pr = calculate_pr_features(pbar, reserve, workload, experiment)
    result = {
        "checkpoint": loaded.checkpoint_path,
        "checkpoint_epoch": int(loaded.checkpoint.get("epoch", -1)),
        "workload_config": workload.path,
        "experiment_config": experiment.path,
        "server_count": int(experiment.server_count),
        "utilization": float(experiment.utilization),
        "Pbar_kw_per_server": float(pbar),
        "R_kw_per_server": float(reserve),
        "Pbar_plus_R": float(pbar + reserve),
        "Pbar_minus_R": float(pbar - reserve),
        "R_over_Pbar": float(reserve / max(pbar, 1e-12)),
        "Pbar_ratio": float(pr["Pbar_ratio"]),
        "R_ratio": float(pr["R_ratio"]),
        "weights": [float(x) for x in weights],
        "Predicted_Mean_Tracking": mean_tracking,
        "Predicted_P90_Tracking": p90_tracking,
        "Predicted_QoS_Probabilities": [float(x) for x in probs],
        "Predicted_Max_Pj": max_pj,
        "Exact_Tracking_Pass": bool(exact_tracking_pass),
        "Exact_QoS_Pass": bool(exact_qos_pass),
        "Exact_Both_Pass": bool(exact_tracking_pass and exact_qos_pass),
        "Safety_Tracking_Pass": bool(safety_tracking_pass),
        "Safety_QoS_Pass": bool(safety_qos_pass),
        "Safety_Both_Pass": bool(safety_tracking_pass and safety_qos_pass),
        "Exact_Tracking_Slack": float(safety.exact_tracking_threshold - p90_tracking),
        "Exact_QoS_Slack": float(safety.exact_qos_threshold - max_pj),
        "Safety_Tracking_Slack": float(safety.selection_tracking_limit - p90_tracking),
        "Safety_QoS_Slack": float(safety.selection_qos_limit - max_pj),
        **costs,
        **{f"Safety_{key}": value for key, value in safety.to_dict().items()},
    }
    per_job = pd.DataFrame(
        {
            "Job_Index": np.arange(workload.job_count),
            "Job_Type": workload.job_names,
            "Weight": np.asarray(weights, dtype=float),
            "Predicted_Pj": probs,
            "Exact_QoS_Pass": probs <= safety.exact_qos_threshold,
            "Safety_QoS_Pass": probs <= safety.selection_qos_limit,
            "Exact_QoS_Slack": safety.exact_qos_threshold - probs,
            "Safety_QoS_Slack": safety.selection_qos_limit - probs,
        }
    )
    return result, per_job


def predict_configuration(
    loaded: LoadedBehaviorModel,
    *,
    workload: WorkloadSpec,
    experiment: ExperimentSpec,
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    weights: Sequence[float],
    safety: SafetyLimits,
    bounds: PRBounds | None = None,
    r_over_p_max: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    weights_array = validate_weights(weights, workload.job_count)
    if bounds is not None:
        validate_physical_bid(pbar_kw_per_server, r_kw_per_server, bounds, r_over_p_max=r_over_p_max)
    _, tokens, global_features, _ = make_feature_row(
        pbar_kw_per_server=pbar_kw_per_server,
        r_kw_per_server=r_kw_per_server,
        weights=weights_array,
        workload=workload,
        experiment=experiment,
    )
    token_scaled, global_scaled = standardize_features(tokens, global_features, loaded.metadata)
    token_tensor = torch.as_tensor(token_scaled, dtype=torch.float32, device=loaded.device).unsqueeze(0)
    global_tensor = torch.as_tensor(global_scaled, dtype=torch.float32, device=loaded.device).unsqueeze(0)
    mask = torch.ones((1, workload.job_count), dtype=torch.bool, device=loaded.device)
    with torch.no_grad():
        output = loaded.model(global_tensor, token_tensor, mask)
    return _prediction_from_tensors(
        loaded=loaded,
        tracking_logs=output["tracking_logs"],
        qos_probabilities=output["qos_probabilities"],
        pbar=float(pbar_kw_per_server),
        reserve=float(r_kw_per_server),
        workload=workload,
        experiment=experiment,
        safety=safety,
        weights=weights_array,
    )


# ---------------------------------------------------------------------------
# Differentiable feature/objective construction for optimization
# ---------------------------------------------------------------------------


def _torch_constants(metadata: BehaviorDataMetadata, device: torch.device) -> dict:
    return {
        "token_mean": torch.tensor(metadata.token_feature_mean, dtype=torch.float32, device=device),
        "token_std": torch.tensor(metadata.token_feature_std, dtype=torch.float32, device=device),
        "global_mean": torch.tensor(metadata.global_feature_mean, dtype=torch.float32, device=device),
        "global_std": torch.tensor(metadata.global_feature_std, dtype=torch.float32, device=device),
    }


def build_differentiable_features(
    *,
    pbar: torch.Tensor,
    reserve: torch.Tensor,
    weights: torch.Tensor,
    workload: WorkloadSpec,
    experiment: ExperimentSpec,
    metadata: BehaviorDataMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pbar.device
    dtype = pbar.dtype
    starts, job_count = weights.shape
    if job_count != workload.job_count:
        raise ValueError("Weight width does not match workload job count")
    mix = torch.tensor(workload.mix, dtype=dtype, device=device)
    pmin_w, pmax_w, tmin, tmax, qos_threshold, job_size = [mix[:, index] for index in range(6)]
    runtime_ratio = tmax / torch.clamp(tmin, min=1e-9)
    qos_headroom = qos_threshold / torch.clamp(runtime_ratio, min=1e-9)
    queue_pressure = experiment.utilization / torch.clamp(job_count * weights, min=1e-6)
    fixed = torch.stack(
        [
            pmin_w / 1000.0,
            pmax_w / 1000.0,
            (pmax_w - pmin_w) / 1000.0,
            torch.log10(torch.clamp(tmin, min=1e-9)),
            torch.log10(torch.clamp(tmax, min=1e-9)),
            runtime_ratio,
            qos_threshold,
            job_size,
            qos_headroom,
        ],
        dim=-1,
    ).unsqueeze(0).expand(starts, -1, -1)
    variable = torch.stack(
        [
            weights,
            weights * job_count,
            weights * float(experiment.server_count) / 1000.0,
            queue_pressure,
        ],
        dim=-1,
    )
    tokens = torch.cat([fixed, variable], dim=-1)

    pbar_denominator, r_denominator = calculate_pr_denominators(workload, experiment)
    p_actual = pbar * 1000.0 * experiment.server_count
    r_actual = reserve * 1000.0 * experiment.server_count
    pbar_ratio = p_actual / pbar_denominator
    r_ratio = r_actual / r_denominator
    job_count_tensor = torch.full_like(pbar, float(job_count))
    global_features = torch.stack(
        [
            pbar,
            reserve,
            pbar - reserve,
            pbar + reserve,
            reserve / torch.clamp(pbar, min=1e-9),
            pbar_ratio,
            r_ratio,
            torch.full_like(pbar, float(experiment.utilization)),
            torch.full_like(pbar, float(experiment.server_count) / 1000.0),
            torch.full_like(pbar, math.log10(max(experiment.server_count, 1))),
            1.0 / job_count_tensor,
            job_count_tensor / 8.0,
        ],
        dim=-1,
    )
    stats = _torch_constants(metadata, device)
    tokens_scaled = (tokens - stats["token_mean"]) / stats["token_std"]
    global_scaled = (global_features - stats["global_mean"]) / stats["global_std"]
    mask = torch.ones((starts, job_count), dtype=torch.bool, device=device)
    return global_scaled, tokens_scaled, mask


def reconstruct_differentiable_outputs(
    *,
    loaded: LoadedBehaviorModel,
    pbar: torch.Tensor,
    reserve: torch.Tensor,
    model_output: dict[str, torch.Tensor],
    experiment: ExperimentSpec,
) -> dict[str, torch.Tensor]:
    constants = loaded.constants
    mean_tracking, p90_tracking = decode_tracking_logs(model_output["tracking_logs"], constants)
    duration = float(experiment.simulation_duration_seconds)
    coefficient = DEFAULT_ENERGY_PRICE_PER_KWH / KWH_IN_WATT_SECONDS
    p_actual = pbar * 1000.0 * experiment.server_count
    r_actual = reserve * 1000.0 * experiment.server_count
    power_cost = coefficient * (p_actual - r_actual) * duration
    m_track = coefficient * r_actual * mean_tracking * duration
    m_rsr = power_cost + m_track
    ctrack = constants.ctrack_psi * F.softplus(constants.ctrack_mu * (p90_tracking - constants.tracking_threshold))
    per_job_cqos = constants.qos_beta * F.softplus(constants.qos_rho * (model_output["qos_probabilities"] - constants.qos_threshold))
    cqos = per_job_cqos.sum(dim=1)
    objective = m_rsr + ctrack + cqos
    return {
        "mean_tracking": mean_tracking,
        "p90_tracking": p90_tracking,
        "qos_probabilities": model_output["qos_probabilities"],
        "max_pj": model_output["qos_probabilities"].max(dim=1).values,
        "power_cost": power_cost,
        "m_track": m_track,
        "m_rsr": m_rsr,
        "ctrack": ctrack,
        "cqos": cqos,
        "objective": objective,
    }


def _logit(value: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    value = torch.clamp(value, eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def parameterize_weights(
    logits: torch.Tensor,
    weight_min: float | None,
    weight_max: float | None = None,
) -> torch.Tensor:
    """Map unconstrained logits to a bounded simplex.

    With only a lower bound, an affine Softmax gives an exact, efficient map.
    With both lower and upper bounds, a differentiable scalar offset is solved
    by Newton iterations so that every weight lies in [lower, upper] and the
    row sums remain one.
    """
    lower = float(weight_min or 0.0)
    upper = float(weight_max if weight_max is not None else 1.0)
    job_count = logits.shape[1]
    if lower < 0 or upper > 1 or upper <= lower:
        raise ValueError(f"Invalid weight bounds: lower={lower}, upper={upper}")
    if job_count * lower >= 1.0 or job_count * upper < 1.0:
        raise ValueError(f"Weight bounds are infeasible for J={job_count}: lower={lower}, upper={upper}")
    if upper >= 1.0 - (job_count - 1) * lower - 1e-12:
        probs = torch.softmax(logits, dim=-1)
        return lower + (1.0 - job_count * lower) * probs

    # Solve sum_i lower + (upper-lower)*sigmoid(z_i + lambda) = 1.
    work = logits.to(dtype=torch.float64)
    lam = torch.zeros((logits.shape[0], 1), dtype=work.dtype, device=work.device)
    scale = upper - lower
    for _ in range(32):
        sig = torch.sigmoid(work + lam)
        residual = lower * job_count + scale * sig.sum(dim=1, keepdim=True) - 1.0
        derivative = scale * (sig * (1.0 - sig)).sum(dim=1, keepdim=True).clamp_min(1e-12)
        lam = lam - residual / derivative
    weights = lower + scale * torch.sigmoid(work + lam)
    return weights.to(dtype=logits.dtype)


def parameterize_bid(
    p_logits: torch.Tensor,
    r_logits: torch.Tensor,
    bounds: PRBounds,
    settings: OptimizationSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_lower = max(bounds.pbar_lower_kw_per_server, settings.pbar_min_override or -math.inf)
    p_upper = min(bounds.pbar_upper_kw_per_server, settings.pbar_max_override or math.inf)
    r_lower = max(bounds.r_lower_kw_per_server, settings.r_min_override or -math.inf)
    p_upper = min(p_upper, bounds.pr_upper_kw_per_server - r_lower)
    if p_upper <= p_lower:
        raise ValueError("Configured Pbar/R bounds leave no feasible Pbar interval")
    pbar = p_lower + torch.sigmoid(p_logits) * (p_upper - p_lower)
    r_upper = torch.minimum(pbar, torch.full_like(pbar, bounds.pr_upper_kw_per_server) - pbar)
    if settings.r_over_p_max is not None:
        r_upper = torch.minimum(r_upper, float(settings.r_over_p_max) * pbar)
    if settings.r_max_override is not None:
        r_upper = torch.minimum(r_upper, torch.full_like(r_upper, float(settings.r_max_override)))
    available = torch.clamp(r_upper - r_lower, min=1e-7)
    reserve = r_lower + torch.sigmoid(r_logits) * available
    return pbar, reserve, r_upper



def prepare_ratio_starts(
    *,
    workload: WorkloadSpec,
    experiment: ExperimentSpec,
    bounds: PRBounds,
    settings: OptimizationSettings,
    ratio_starts: Sequence[Sequence[float]],
) -> pd.DataFrame:
    """Convert normalized P/R starts to legal physical starts.

    This is the generic, non-monkey-patched implementation of the exact-start
    logic in the original paired-comparison notebook.
    """
    p_den, r_den = calculate_pr_denominators(workload, experiment)
    p_lower = max(
        bounds.pbar_lower_kw_per_server,
        float(settings.pbar_min_override) if settings.pbar_min_override is not None else -math.inf,
    )
    r_lower = max(
        bounds.r_lower_kw_per_server,
        float(settings.r_min_override) if settings.r_min_override is not None else -math.inf,
    )
    p_upper = min(
        bounds.pbar_upper_kw_per_server,
        bounds.pr_upper_kw_per_server - r_lower,
        float(settings.pbar_max_override) if settings.pbar_max_override is not None else math.inf,
    )
    eps = 1e-7
    rows: list[dict] = []
    for index, pair in enumerate(ratio_starts):
        p_ratio, r_ratio = map(float, pair)
        requested_p = p_ratio * p_den / (1000.0 * experiment.server_count)
        requested_r = r_ratio * r_den / (1000.0 * experiment.server_count)
        effective_p, effective_r = requested_p, requested_r
        projected = False
        skipped = False
        reason = ""
        try:
            validate_physical_bid(
                effective_p, effective_r, bounds,
                r_over_p_max=settings.r_over_p_max,
            )
        except Exception as exc:
            reason = str(exc)
            if settings.illegal_start_policy == "strict":
                raise
            if settings.illegal_start_policy == "skip":
                skipped = True
            else:
                effective_p = float(np.clip(requested_p, p_lower + eps, p_upper - eps))
                r_upper = min(
                    effective_p,
                    bounds.pr_upper_kw_per_server - effective_p,
                    float(settings.r_over_p_max) * effective_p
                    if settings.r_over_p_max is not None else math.inf,
                    float(settings.r_max_override)
                    if settings.r_max_override is not None else math.inf,
                )
                if r_upper <= r_lower:
                    raise ValueError(f"No legal reserve interval for requested start {pair}")
                effective_r = float(np.clip(requested_r, r_lower + eps, r_upper - eps))
                validate_physical_bid(
                    effective_p, effective_r, bounds,
                    r_over_p_max=settings.r_over_p_max,
                )
                projected = True
        record = {
            "Start_Index": int(index),
            "Start_ID": f"P{p_ratio:g}_R{r_ratio:g}",
            "Requested_Pbar_Ratio": p_ratio,
            "Requested_R_Ratio": r_ratio,
            "Requested_Pbar": requested_p,
            "Requested_R": requested_r,
            "Skipped": bool(skipped),
            "Projected": bool(projected),
            "Projection_Reason": reason,
        }
        if not skipped:
            features = calculate_pr_features(
                effective_p, effective_r, workload, experiment
            )
            record.update({
                "Effective_Pbar": effective_p,
                "Effective_R": effective_r,
                "Effective_Pbar_Ratio": float(features["Pbar_ratio"]),
                "Effective_R_Ratio": float(features["R_ratio"]),
            })
        rows.append(record)
    manifest = pd.DataFrame(rows)
    if len(manifest) and not (~manifest["Skipped"]).any():
        raise RuntimeError("No legal explicit optimization starts remain")
    return manifest


def _apply_explicit_start_logits(
    *,
    p_logits: torch.Tensor,
    r_logits: torch.Tensor,
    weight_logits: torch.Tensor,
    manifest: pd.DataFrame,
    settings: OptimizationSettings,
    bounds: PRBounds,
    initial_weights: Sequence[float] | None,
    workload: WorkloadSpec,
    weight_bounds: WeightBounds,
) -> None:
    usable = manifest[~manifest["Skipped"].astype(bool)].reset_index(drop=True)
    if len(usable) > len(p_logits):
        raise ValueError("More usable explicit starts than allocated optimizer starts")
    p_lower = max(
        bounds.pbar_lower_kw_per_server,
        float(settings.pbar_min_override) if settings.pbar_min_override is not None else -math.inf,
    )
    r_lower = max(
        bounds.r_lower_kw_per_server,
        float(settings.r_min_override) if settings.r_min_override is not None else -math.inf,
    )
    p_upper = min(
        bounds.pbar_upper_kw_per_server,
        bounds.pr_upper_kw_per_server - r_lower,
        float(settings.pbar_max_override) if settings.pbar_max_override is not None else math.inf,
    )
    equal_or_initial = (
        validate_weights(initial_weights, workload.job_count)
        if initial_weights is not None
        else np.full(workload.job_count, 1.0 / workload.job_count, dtype=float)
    )
    validate_weight_bounds(equal_or_initial, weight_bounds)
    lower, upper = weight_bounds.final_lower, weight_bounds.final_upper
    if upper >= 1.0 - (workload.job_count - 1) * lower - 1e-12:
        base = (equal_or_initial - lower) / max(1.0 - workload.job_count * lower, 1e-9)
        base = np.clip(base, 1e-8, None)
        base = base / base.sum()
        explicit_weight_logits = torch.log(torch.tensor(base, dtype=torch.float32, device=weight_logits.device))
    else:
        scaled = np.clip((equal_or_initial - lower) / (upper - lower), 1e-6, 1 - 1e-6)
        explicit_weight_logits = torch.logit(torch.tensor(scaled, dtype=torch.float32, device=weight_logits.device))
    with torch.no_grad():
        for index, row in usable.iterrows():
            pbar = float(row["Effective_Pbar"])
            reserve = float(row["Effective_R"])
            p_fraction = np.clip((pbar - p_lower) / max(p_upper - p_lower, 1e-9), 1e-6, 1 - 1e-6)
            r_upper = min(
                pbar,
                bounds.pr_upper_kw_per_server - pbar,
                float(settings.r_over_p_max) * pbar
                if settings.r_over_p_max is not None else math.inf,
                float(settings.r_max_override)
                if settings.r_max_override is not None else math.inf,
            )
            r_fraction = np.clip((reserve - r_lower) / max(r_upper - r_lower, 1e-9), 1e-6, 1 - 1e-6)
            p_logits[index].copy_(_logit(torch.tensor(p_fraction, device=p_logits.device)))
            r_logits[index].copy_(_logit(torch.tensor(r_fraction, device=r_logits.device)))
            weight_logits[index].copy_(explicit_weight_logits)


def _initial_logits(
    *,
    settings: OptimizationSettings,
    bounds: PRBounds,
    workload: WorkloadSpec,
    initial_pbar: float | None,
    initial_reserve: float | None,
    initial_weights: Sequence[float] | None,
    weight_bounds: WeightBounds,
    device: torch.device,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(settings.random_seed))
    p_fraction = 0.05 + 0.90 * torch.rand(settings.starts, generator=generator)
    r_fraction = 0.05 + 0.90 * torch.rand(settings.starts, generator=generator)
    weight_logits = torch.randn((settings.starts, workload.job_count), generator=generator)

    # Deliberately seed several useful but still general regions.  This is not
    # a workload-specific answer: it prevents all starts from being strongly
    # skewed and gives the optimizer both broad and conservative initial basins.
    near_equal_count = min(settings.starts, int(round(settings.starts * settings.near_equal_start_fraction)))
    if near_equal_count > 0:
        weight_logits[:near_equal_count] = 0.25 * torch.randn(
            (near_equal_count, workload.job_count), generator=generator
        )
    high_p_low_r_count = min(settings.starts, int(round(settings.starts * settings.high_p_low_r_start_fraction)))
    if high_p_low_r_count > 0:
        p_fraction[:high_p_low_r_count] = 0.70 + 0.25 * torch.rand(high_p_low_r_count, generator=generator)
        r_fraction[:high_p_low_r_count] = 0.05 + 0.35 * torch.rand(high_p_low_r_count, generator=generator)
    p_logits = _logit(p_fraction)
    r_logits = _logit(r_fraction)
    if initial_pbar is not None:
        p_lower = max(bounds.pbar_lower_kw_per_server, settings.pbar_min_override or -math.inf)
        p_upper = min(bounds.pbar_upper_kw_per_server, settings.pbar_max_override or math.inf)
        r_lower = max(bounds.r_lower_kw_per_server, settings.r_min_override or -math.inf)
        p_upper = min(p_upper, bounds.pr_upper_kw_per_server - r_lower)
        fraction = (float(initial_pbar) - p_lower) / (p_upper - p_lower)
        p_logits[0] = _logit(torch.tensor(fraction))
        with torch.no_grad():
            p_temp, _, r_upper = parameterize_bid(p_logits[:1], r_logits[:1], bounds, settings)
        if initial_reserve is not None:
            r_fraction_value = (float(initial_reserve) - r_lower) / max(float(r_upper[0]) - r_lower, 1e-9)
            r_logits[0] = _logit(torch.tensor(r_fraction_value))
    if initial_weights is not None:
        weights = validate_weights(initial_weights, workload.job_count)
        validate_weight_bounds(weights, weight_bounds)
        lower = weight_bounds.final_lower
        upper = weight_bounds.final_upper
        if upper >= 1.0 - (workload.job_count - 1) * lower - 1e-12:
            base = (weights - lower) / max(1.0 - workload.job_count * lower, 1e-9)
            base = np.clip(base, 1e-8, None)
            base = base / base.sum()
            weight_logits[0] = torch.log(torch.tensor(base, dtype=torch.float32))
        else:
            scaled = np.clip((weights - lower) / (upper - lower), 1e-6, 1.0 - 1e-6)
            weight_logits[0] = torch.logit(torch.tensor(scaled, dtype=torch.float32))
    else:
        weight_logits[0] = 0.0  # equal-weight anchor
    return (
        torch.nn.Parameter(p_logits.to(device=device, dtype=torch.float32)),
        torch.nn.Parameter(r_logits.to(device=device, dtype=torch.float32)),
        torch.nn.Parameter(weight_logits.to(device=device, dtype=torch.float32)),
    )


def _cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    if total_steps <= 1:
        return min_lr
    progress = step / float(total_steps - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _candidate_distance(
    candidate: pd.Series,
    accepted: pd.Series,
    bounds: PRBounds,
    job_count: int,
) -> float:
    p_scale = max(bounds.pbar_upper_kw_per_server - bounds.pbar_lower_kw_per_server, 1e-9)
    r_scale = max(bounds.pr_upper_kw_per_server, 1e-9)
    a_weights = np.asarray(candidate["weights"], dtype=float)
    b_weights = np.asarray(accepted["weights"], dtype=float)
    vector_a = np.concatenate(
        [
            [(candidate["Pbar_kw_per_server"] - bounds.pbar_lower_kw_per_server) / p_scale],
            [candidate["R_kw_per_server"] / r_scale],
            a_weights * math.sqrt(job_count),
        ]
    )
    vector_b = np.concatenate(
        [
            [(accepted["Pbar_kw_per_server"] - bounds.pbar_lower_kw_per_server) / p_scale],
            [accepted["R_kw_per_server"] / r_scale],
            b_weights * math.sqrt(job_count),
        ]
    )
    return float(np.linalg.norm(vector_a - vector_b))


def select_distinct_top_k(
    candidates: pd.DataFrame,
    *,
    bounds: PRBounds,
    job_count: int,
    top_k: int,
    minimum_distance: float,
    feasibility_column: str = "Safety_Both_Pass",
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    pool = candidates[candidates[feasibility_column].astype(bool)].sort_values(
        ["Predicted_Full_Objective", "Safety_Tracking_Slack", "Safety_QoS_Slack"],
        ascending=[True, False, False],
    )
    accepted_rows: list[pd.Series] = []
    for _, row in pool.iterrows():
        if all(
            _candidate_distance(row, prior, bounds, job_count) >= minimum_distance
            for prior in accepted_rows
        ):
            accepted_rows.append(row)
        if len(accepted_rows) >= top_k:
            break
    if not accepted_rows:
        return pool.iloc[0:0].copy()
    output = pd.DataFrame(accepted_rows).reset_index(drop=True)
    output.insert(0, "Candidate_Rank", np.arange(1, len(output) + 1))
    return output


def optimize_candidates(
    loaded: LoadedBehaviorModel,
    *,
    workload: WorkloadSpec,
    experiment: ExperimentSpec,
    bounds: PRBounds,
    safety: SafetyLimits,
    settings: OptimizationSettings,
    initial_pbar: float | None = None,
    initial_reserve: float | None = None,
    initial_weights: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings.validate(workload.job_count)
    weight_bounds = resolve_effective_weight_bounds(
        settings,
        job_count=workload.job_count,
        server_count=experiment.server_count,
    )
    p_logits, r_logits, weight_logits = _initial_logits(
        settings=settings,
        bounds=bounds,
        workload=workload,
        initial_pbar=initial_pbar,
        initial_reserve=initial_reserve,
        initial_weights=initial_weights,
        weight_bounds=weight_bounds,
        device=loaded.device,
    )
    explicit_manifest = pd.DataFrame()
    if settings.explicit_ratio_starts:
        explicit_manifest = prepare_ratio_starts(
            workload=workload,
            experiment=experiment,
            bounds=bounds,
            settings=settings,
            ratio_starts=settings.explicit_ratio_starts,
        )
        _apply_explicit_start_logits(
            p_logits=p_logits,
            r_logits=r_logits,
            weight_logits=weight_logits,
            manifest=explicit_manifest,
            settings=settings,
            bounds=bounds,
            initial_weights=initial_weights,
            workload=workload,
            weight_bounds=weight_bounds,
        )
    optimizer = torch.optim.Adam([p_logits, r_logits, weight_logits], lr=settings.learning_rate)
    trajectory_rows: list[dict] = []
    loaded.model.eval()
    for step in range(settings.iterations):
        lr = _cosine_lr(step, settings.iterations, settings.learning_rate, settings.minimum_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        pbar, reserve, r_upper = parameterize_bid(p_logits, r_logits, bounds, settings)
        weights = parameterize_weights(weight_logits, weight_bounds.final_lower, weight_bounds.final_upper)
        global_features, token_features, mask = build_differentiable_features(
            pbar=pbar,
            reserve=reserve,
            weights=weights,
            workload=workload,
            experiment=experiment,
            metadata=loaded.metadata,
        )
        output = loaded.model(global_features, token_features, mask)
        reconstructed = reconstruct_differentiable_outputs(
            loaded=loaded,
            pbar=pbar,
            reserve=reserve,
            model_output=output,
            experiment=experiment,
        )
        if settings.mode == "pure_objective":
            tracking_limit = loaded.constants.tracking_threshold
            qos_limit = loaded.constants.qos_threshold
            penalty_multiplier = 0.0
        elif settings.mode == "exact_constrained":
            tracking_limit = loaded.constants.tracking_threshold
            qos_limit = loaded.constants.qos_threshold
            ramp_steps = max(1, int(settings.iterations * settings.penalty_ramp_fraction))
            penalty_multiplier = min(1.0, (step + 1) / ramp_steps)
        else:
            tracking_limit = safety.selection_tracking_limit
            qos_limit = safety.selection_qos_limit
            ramp_steps = max(1, int(settings.iterations * settings.penalty_ramp_fraction))
            penalty_multiplier = min(1.0, (step + 1) / ramp_steps)
        tracking_violation = F.relu(reconstructed["p90_tracking"] - tracking_limit)
        qos_violation = F.relu(reconstructed["qos_probabilities"] - qos_limit)
        constraint_penalty = (
            settings.tracking_penalty * tracking_violation.square()
            + settings.qos_penalty * qos_violation.square().sum(dim=1)
        )
        loss_per_start = reconstructed["objective"] + penalty_multiplier * constraint_penalty
        loss = loss_per_start.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Optimization loss became non-finite at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p_logits, r_logits, weight_logits], max_norm=10.0)
        optimizer.step()

        if step % max(settings.log_every, 1) == 0 or step == settings.iterations - 1:
            with torch.no_grad():
                safety_pass = (
                    (reconstructed["p90_tracking"] <= safety.selection_tracking_limit)
                    & (reconstructed["max_pj"] <= safety.selection_qos_limit)
                )
                exact_pass = (
                    (reconstructed["p90_tracking"] <= loaded.constants.tracking_threshold)
                    & (reconstructed["max_pj"] <= loaded.constants.qos_threshold)
                )
                trajectory_rows.append(
                    {
                        "Iteration": int(step),
                        "Learning_Rate": float(lr),
                        "Best_Penalized_Loss": float(loss_per_start.min().detach().cpu()),
                        "Best_Predicted_Objective": float(reconstructed["objective"].min().detach().cpu()),
                        "Safety_Feasible_Starts": int(safety_pass.sum().detach().cpu()),
                        "Exact_Feasible_Starts": int(exact_pass.sum().detach().cpu()),
                        "Median_P90": float(reconstructed["p90_tracking"].median().detach().cpu()),
                        "Median_Max_Pj": float(reconstructed["max_pj"].median().detach().cpu()),
                        "Penalty_Multiplier": float(penalty_multiplier),
                    }
                )

    with torch.no_grad():
        pbar, reserve, r_upper = parameterize_bid(p_logits, r_logits, bounds, settings)
        weights = parameterize_weights(weight_logits, weight_bounds.final_lower, weight_bounds.final_upper)
        global_features, token_features, mask = build_differentiable_features(
            pbar=pbar,
            reserve=reserve,
            weights=weights,
            workload=workload,
            experiment=experiment,
            metadata=loaded.metadata,
        )
        output = loaded.model(global_features, token_features, mask)
        reconstructed = reconstruct_differentiable_outputs(
            loaded=loaded,
            pbar=pbar,
            reserve=reserve,
            model_output=output,
            experiment=experiment,
        )
    candidate_rows: list[dict] = []
    for index in range(settings.starts):
        probs = reconstructed["qos_probabilities"][index, : workload.job_count].detach().cpu().numpy().astype(float)
        weight_values = weights[index].detach().cpu().numpy().astype(float)
        p_value = float(pbar[index].detach().cpu())
        r_value = float(reserve[index].detach().cpu())
        max_pj = float(np.max(probs))
        exact_tracking = float(reconstructed["p90_tracking"][index].detach().cpu()) <= loaded.constants.tracking_threshold
        exact_qos = max_pj <= loaded.constants.qos_threshold
        safety_tracking = float(reconstructed["p90_tracking"][index].detach().cpu()) <= safety.selection_tracking_limit
        safety_qos = max_pj <= safety.selection_qos_limit
        weight_bounds_ok = bool(
            float(np.min(weight_values)) >= weight_bounds.final_lower - 1e-6
            and float(np.max(weight_values)) <= weight_bounds.final_upper + 1e-6
            and abs(float(np.sum(weight_values)) - 1.0) <= 1e-5
        )
        candidate_rows.append(
            {
                "Start_Index": int(index),
                "Pbar_kw_per_server": p_value,
                "R_kw_per_server": r_value,
                "Pbar_plus_R": p_value + r_value,
                "Pbar_minus_R": p_value - r_value,
                "R_over_Pbar": r_value / max(p_value, 1e-12),
                "weights": [float(x) for x in weight_values],
                "Predicted_Mean_Tracking": float(reconstructed["mean_tracking"][index].detach().cpu()),
                "Predicted_P90_Tracking": float(reconstructed["p90_tracking"][index].detach().cpu()),
                "Predicted_QoS_Probabilities": [float(x) for x in probs],
                "Predicted_Max_Pj": max_pj,
                "Predicted_Power_Cost": float(reconstructed["power_cost"][index].detach().cpu()),
                "Predicted_Mtrack_Cost": float(reconstructed["m_track"][index].detach().cpu()),
                "Predicted_M_RSR": float(reconstructed["m_rsr"][index].detach().cpu()),
                "Predicted_Ctrack": float(reconstructed["ctrack"][index].detach().cpu()),
                "Predicted_CQoS": float(reconstructed["cqos"][index].detach().cpu()),
                "Predicted_Full_Objective": float(reconstructed["objective"][index].detach().cpu()),
                "Exact_Tracking_Pass": bool(exact_tracking),
                "Exact_QoS_Pass": bool(exact_qos),
                "Exact_Both_Pass": bool(exact_tracking and exact_qos and weight_bounds_ok),
                "Safety_Tracking_Pass": bool(safety_tracking),
                "Safety_QoS_Pass": bool(safety_qos),
                "Safety_Both_Pass": bool(safety_tracking and safety_qos and weight_bounds_ok),
                "Exact_Tracking_Slack": float(loaded.constants.tracking_threshold - reconstructed["p90_tracking"][index].detach().cpu()),
                "Exact_QoS_Slack": float(loaded.constants.qos_threshold - max_pj),
                "Safety_Tracking_Slack": float(safety.selection_tracking_limit - reconstructed["p90_tracking"][index].detach().cpu()),
                "Safety_QoS_Slack": float(safety.selection_qos_limit - max_pj),
                "Weight_Min": float(np.min(weight_values)),
                "Weight_Max": float(np.max(weight_values)),
                "Weight_Bounds_Pass": bool(weight_bounds_ok),
                "Effective_Weight_Min": float(weight_bounds.final_lower),
                "Effective_Weight_Max": float(weight_bounds.final_upper),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    if len(explicit_manifest):
        explicit_usable = explicit_manifest[~explicit_manifest["Skipped"].astype(bool)].reset_index(drop=True)
        # Explicit starts occupy the first N optimizer slots after skipped starts
        # are removed. Preserve their requested/effective ratio provenance.
        explicit_usable = explicit_usable.copy()
        explicit_usable["Start_Index"] = np.arange(len(explicit_usable), dtype=int)
        candidates = candidates.merge(explicit_usable, on="Start_Index", how="left")
    candidates = candidates.sort_values("Predicted_Full_Objective").reset_index(drop=True)
    top_k = select_distinct_top_k(
        candidates,
        bounds=bounds,
        job_count=workload.job_count,
        top_k=settings.top_k,
        minimum_distance=settings.candidate_distance,
        feasibility_column="Safety_Both_Pass" if settings.mode == "margin_constrained" else "Exact_Both_Pass",
    )
    return candidates, top_k, pd.DataFrame(trajectory_rows)


# ---------------------------------------------------------------------------
# Margin calibration and checkpoint cross-scoring
# ---------------------------------------------------------------------------


def margin_calibration_table(
    predictions_csv: str | Path,
    *,
    tracking_margins: Sequence[float] = (0.0, 0.02, 0.04, 0.06, 0.08),
    qos_margins: Sequence[float] = (0.0, 0.005, 0.01, 0.02),
    exact_tracking_threshold: float = 0.3,
    exact_qos_threshold: float = 0.1,
) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    required = ["Actual_P90_Tracking", "Predicted_P90_Tracking", "Actual_Max_Pj", "Predicted_Max_Pj"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Prediction CSV missing columns: {missing}")
    actual = (df["Actual_P90_Tracking"] <= exact_tracking_threshold) & (df["Actual_Max_Pj"] <= exact_qos_threshold)
    rows: list[dict] = []
    for tracking_margin in tracking_margins:
        for qos_margin in qos_margins:
            tracking_limit = exact_tracking_threshold - float(tracking_margin)
            qos_limit = exact_qos_threshold - float(qos_margin)
            predicted = (df["Predicted_P90_Tracking"] <= tracking_limit) & (df["Predicted_Max_Pj"] <= qos_limit)
            tp = int((actual & predicted).sum())
            fp = int((~actual & predicted).sum())
            fn = int((actual & ~predicted).sum())
            tn = int((~actual & ~predicted).sum())
            precision = tp / (tp + fp) if tp + fp else np.nan
            recall = tp / (tp + fn) if tp + fn else np.nan
            rows.append(
                {
                    "Tracking_Margin": float(tracking_margin),
                    "QoS_Margin": float(qos_margin),
                    "Tracking_Limit": float(tracking_limit),
                    "QoS_Limit": float(qos_limit),
                    "Predicted_Feasible": int(predicted.sum()),
                    "True_Positive": tp,
                    "False_Positive": fp,
                    "False_Negative": fn,
                    "True_Negative": tn,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
                    "False_Feasible_Rate": fp / max(int((~actual).sum()), 1),
                }
            )
    return pd.DataFrame(rows).sort_values(["False_Feasible_Rate", "Recall"], ascending=[True, False]).reset_index(drop=True)


def score_candidate_table_with_checkpoint(
    candidates: pd.DataFrame,
    *,
    checkpoint_path: str | Path,
    workload: WorkloadSpec,
    experiment: ExperimentSpec,
    safety: SafetyLimits,
    device_name: str = "auto",
    prefix: str = "Secondary",
) -> pd.DataFrame:
    loaded = load_behavior_model(checkpoint_path, device_name=device_name)
    output = candidates.copy()
    p90_values = []
    max_pj_values = []
    objective_values = []
    pass_values = []
    for _, row in candidates.iterrows():
        result, _ = predict_configuration(
            loaded,
            workload=workload,
            experiment=experiment,
            pbar_kw_per_server=float(row["Pbar_kw_per_server"]),
            r_kw_per_server=float(row["R_kw_per_server"]),
            weights=row["weights"] if isinstance(row["weights"], list) else parse_list(row["weights"], "weights"),
            safety=safety,
        )
        p90_values.append(result["Predicted_P90_Tracking"])
        max_pj_values.append(result["Predicted_Max_Pj"])
        objective_values.append(result["Predicted_Full_Objective"])
        pass_values.append(result["Safety_Both_Pass"])
    output[f"{prefix}_P90"] = p90_values
    output[f"{prefix}_Max_Pj"] = max_pj_values
    output[f"{prefix}_Objective"] = objective_values
    output[f"{prefix}_Safety_Pass"] = pass_values
    return output


# ---------------------------------------------------------------------------
# FlexDC validation helpers
# ---------------------------------------------------------------------------


def build_flexdc_wizard_command(
    *,
    python_executable: str,
    flexdc_root: str | Path,
    gradient_config: str | Path,
    experiment_config: str | Path,
    cluster_config: str | Path,
    workload_config: str | Path,
    output_label: str,
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    weights: Sequence[float],
    utilization: float,
    policy_name: str = "AQA",
    node_count_control: bool = True,
    pbar_lower_factor: float = 0.9,
    pbar_upper_factor: float = 1.0,
    pr_upper_factor: float = 1.2,
) -> tuple[list[str], Path, dict]:
    flexdc_root = Path(flexdc_root).resolve()
    peacsim_dir = flexdc_root / "src" / "peacsim"
    wizard = peacsim_dir / "am_data_extraction_wizard.py"
    if not wizard.exists():
        raise FileNotFoundError(wizard)
    command = [
        str(python_executable),
        "-u",
        wizard.name,
        "--gradient-config",
        str(Path(gradient_config).resolve()),
        "--experiment-config",
        str(Path(experiment_config).resolve()),
        "--cluster-config",
        str(Path(cluster_config).resolve()),
        "--policy-name",
        str(policy_name),
        "--job-config",
        str(Path(workload_config).resolve()),
        "--output-dir",
        str(output_label),
        "--utilization-values",
        str(float(utilization)),
        "--auto-workload-pr-sweep",
        "false",
        "--pbar-kw-per-server-values",
        f"{float(pbar_kw_per_server):.12g}",
        "--r-kw-per-server-values",
        f"{float(r_kw_per_server):.12g}",
        "--weight-vectors",
        ",".join(f"{float(value):.15g}" for value in weights),
        "--node-count-control",
        "true" if node_count_control else "false",
        "--pbar-lower-factor",
        str(float(pbar_lower_factor)),
        "--pbar-upper-factor",
        str(float(pbar_upper_factor)),
        "--pr-upper-factor",
        str(float(pr_upper_factor)),
        "--pr-chunk-index",
        "0",
        "--pr-num-chunks",
        "1",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(flexdc_root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    return command, peacsim_dir, environment


def find_latest_wizard_output(peacsim_dir: str | Path, output_label: str, *, after_time: float | None = None) -> Path:
    base = Path(peacsim_dir) / "output" / "optimization"
    matches = list(base.glob(f"{output_label}_*"))
    if after_time is not None:
        matches = [path for path in matches if path.stat().st_mtime >= after_time - 2.0]
    if not matches:
        raise FileNotFoundError(f"No FlexDC output found for label {output_label} under {base}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def parse_probability_vector(value) -> np.ndarray:
    return np.asarray(parse_list(value, "QoS_Delay_Probabilities"), dtype=np.float64)


def parse_flexdc_validation_output(
    output_dir: str | Path,
    *,
    constants: FlexDCBehaviorConstants,
) -> tuple[dict, pd.DataFrame]:
    output_dir = Path(output_dir)
    results_path = output_dir / "grid_search_results.csv"
    diagnostics_path = output_dir / "grid_search_diagnostics.csv"
    if not results_path.exists() or not diagnostics_path.exists():
        raise FileNotFoundError(f"Expected results and diagnostics in {output_dir}")
    results = pd.read_csv(results_path)
    diagnostics = pd.read_csv(diagnostics_path)
    if len(results) != 1 or len(diagnostics) != 1:
        raise ValueError(f"Expected one validation row; got results={len(results)}, diagnostics={len(diagnostics)}")
    result_row = results.iloc[0]
    diagnostic_row = diagnostics.iloc[0]
    if int(result_row["Iteration"]) != int(diagnostic_row["Iteration"]):
        raise ValueError("Results/diagnostics Iteration mismatch")
    if int(result_row.get("Weight_Sample_ID", 0)) != int(diagnostic_row.get("Weight_Sample_ID", 0)):
        raise ValueError("Results/diagnostics Weight_Sample_ID mismatch")
    row = result_row
    probs = parse_probability_vector(row["QoS_Delay_Probabilities"])
    mean_tracking = float(row["Mtrack_Error_MeanAbs_Normalized"])
    p90_tracking = float(diagnostic_row["Ctrack_Epsilon_90th"])
    power_cost = float(row["Simulator_Power_Cost"])
    mtrack = float(row["Mtrack_Cost"])
    m_rsr = power_cost + mtrack
    ctrack = float(constants.ctrack_psi * np.logaddexp(0.0, constants.ctrack_mu * (p90_tracking - constants.tracking_threshold)))
    cqos = float(constants.qos_beta * np.logaddexp(0.0, constants.qos_rho * (probs - constants.qos_threshold)).sum())
    weight_columns = sorted(
        [column for column in results.columns if re.fullmatch(r"Weight_\d+", str(column))],
        key=lambda column: int(str(column).split("_")[-1]),
    )
    weights = [float(row[column]) for column in weight_columns]
    summary = {
        "FlexDC_Output_Dir": str(output_dir),
        "FlexDC_Results_CSV": str(results_path),
        "FlexDC_Diagnostics_CSV": str(diagnostics_path),
        "Actual_Pbar_kw_per_server": float(row["Pbar_kw_per_server"]),
        "Actual_R_kw_per_server": float(row["R_kw_per_server"]),
        "Actual_Weights": weights,
        "Actual_Mean_Tracking": mean_tracking,
        "Actual_P90_Tracking": p90_tracking,
        "Actual_QoS_Probabilities": [float(value) for value in probs],
        "Actual_Max_Pj": float(np.max(probs)),
        "Actual_QoS_Violation_Ratio": float(row.get("QoS_Violation_Ratio", np.nan)),
        "Actual_Power_Cost": power_cost,
        "Actual_Mtrack_Cost": mtrack,
        "Actual_M_RSR": m_rsr,
        "Actual_Ctrack": ctrack,
        "Actual_CQoS": cqos,
        "Actual_Full_Objective": m_rsr + ctrack + cqos,
        "Actual_Tracking_Pass": bool(p90_tracking <= constants.tracking_threshold),
        "Actual_QoS_Pass": bool(float(np.max(probs)) <= constants.qos_threshold),
        "Actual_Both_Pass": bool(p90_tracking <= constants.tracking_threshold and float(np.max(probs)) <= constants.qos_threshold),
    }
    per_job = pd.DataFrame(
        {
            "Job_Index": np.arange(len(probs)),
            "Actual_Pj": probs,
            "Actual_QoS_Pass": probs <= constants.qos_threshold,
            "Actual_QoS_Slack": constants.qos_threshold - probs,
        }
    )
    return summary, per_job


def run_flexdc_validation(
    *,
    python_executable: str,
    flexdc_root: str | Path,
    gradient_config: str | Path,
    experiment_config: str | Path,
    cluster_config: str | Path,
    workload_config: str | Path,
    output_label: str,
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    weights: Sequence[float],
    utilization: float,
    constants: FlexDCBehaviorConstants,
    policy_name: str = "AQA",
    node_count_control: bool = True,
    timeout_seconds: int = 1800,
    dry_run: bool = False,
) -> tuple[dict, pd.DataFrame]:
    workload_spec = read_workload_config(workload_config)
    experiment_spec = read_experiment_config(
        experiment_config,
        utilization_override=float(utilization),
    )
    wizard_weight_bounds = calculate_weight_bounds(
        workload_spec.job_count,
        experiment_spec.server_count,
    )
    validate_weights(weights, workload_spec.job_count)
    validate_weight_bounds(weights, wizard_weight_bounds)

    command, cwd, environment = build_flexdc_wizard_command(
        python_executable=python_executable,
        flexdc_root=flexdc_root,
        gradient_config=gradient_config,
        experiment_config=experiment_config,
        cluster_config=cluster_config,
        workload_config=workload_config,
        output_label=output_label,
        pbar_kw_per_server=pbar_kw_per_server,
        r_kw_per_server=r_kw_per_server,
        weights=weights,
        utilization=utilization,
        policy_name=policy_name,
        node_count_control=node_count_control,
    )
    if dry_run:
        return {
            "Dry_Run": True,
            "Command": command,
            "CWD": str(cwd),
            "PYTHONPATH": environment.get("PYTHONPATH", ""),
        }, pd.DataFrame()
    start_time = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FlexDC validation failed\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout[-8000:]}\n"
            f"STDERR:\n{completed.stderr[-8000:]}"
        )
    output_dir = find_latest_wizard_output(cwd, output_label, after_time=start_time)
    summary, per_job = parse_flexdc_validation_output(output_dir, constants=constants)
    summary["FlexDC_Command"] = command
    summary["FlexDC_STDOUT_Tail"] = completed.stdout[-4000:]
    summary["FlexDC_STDERR_Tail"] = completed.stderr[-4000:]
    return summary, per_job


def dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in output.columns:
        if len(output) and output[column].map(lambda value: isinstance(value, (list, tuple, dict, np.ndarray))).any():
            output[column] = output[column].map(
                lambda value: json.dumps(value.tolist() if isinstance(value, np.ndarray) else value)
                if isinstance(value, (list, tuple, dict, np.ndarray))
                else value
            )
    return output


def write_json(path: str | Path, payload: dict | list) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)))
    return path
