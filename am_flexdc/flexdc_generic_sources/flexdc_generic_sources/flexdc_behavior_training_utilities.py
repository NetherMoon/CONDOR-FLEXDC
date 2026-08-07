"""Training utilities for CONDOR-based the FlexDC behavior model.

Key improvements over v1
------------------------
* physics-derived per-job and global features, standardized on training only;
* preassigned train/validation/test splits with seed replicas kept together;
* backward-compatible grouped split fallback for older datasets;
* base-configuration-aware sampling so repeated seeds do not dominate;
* configurable context/feasibility/boundary-balanced training sampler;
* QoS-emphasized masked loss and optional boundary weighting;
* warmup + cosine learning-rate schedule, gradient clipping, early stopping;
* latest/best-loss/best-objective/best-feasibility checkpoints every epoch;
* resumable optimizer state and W&B run metadata;
* per-workload feasibility and boundary metrics so global accuracy cannot hide
  a complete W2 failure.

The neural model directly predicts only log tracking values and P_j.  Known
FlexDC monetary costs are reconstructed analytically.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


MEAN_TRACKING_LOG_FLOOR = 1e-6
P90_TRACKING_LOG_FLOOR = 1e-3
TRACKING_THRESHOLD = 0.3
QOS_THRESHOLD = 0.1
CTRACK_PSI = 1.0
CTRACK_MU = 10.0
QOS_BETA = 20.0
QOS_RHO = 2.0

TOKEN_FEATURE_NAMES = [
    "pmin_kw",
    "pmax_kw",
    "power_range_kw",
    "log10_tmin",
    "log10_tmax",
    "runtime_ratio",
    "qos_threshold",
    "job_size",
    "qos_headroom",
    "weight",
    "weight_relative_to_equal",
    "allocated_nodes_per_1000",
    "queue_pressure",
]

GLOBAL_FEATURE_NAMES = [
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "Pbar_minus_R",
    "Pbar_plus_R",
    "R_over_Pbar",
    "Pbar_ratio",
    "R_ratio",
    "utilization",
    "server_count_per_1000",
    "log10_server_count",
    "inverse_job_count",
    "job_count_per_8",
]

MERGE_KEYS_PRIORITY = [
    ["Plan_Row_ID"],
    ["Source_Output_Dir", "Iteration"],
    ["Source_Output_Dir", "Iteration", "Weight_Sample_ID"],
]


@dataclass(frozen=True)
class FlexDCBehaviorConstants:
    mean_tracking_log_floor: float = MEAN_TRACKING_LOG_FLOOR
    p90_tracking_log_floor: float = P90_TRACKING_LOG_FLOOR
    tracking_threshold: float = TRACKING_THRESHOLD
    qos_threshold: float = QOS_THRESHOLD
    ctrack_psi: float = CTRACK_PSI
    ctrack_mu: float = CTRACK_MU
    qos_beta: float = QOS_BETA
    qos_rho: float = QOS_RHO

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BehaviorDataMetadata:
    token_feature_names: list[str]
    token_feature_mean: list[float]
    token_feature_std: list[float]
    global_feature_names: list[str]
    global_feature_mean: list[float]
    global_feature_std: list[float]
    log_mean_tracking_mean: float
    log_mean_tracking_std: float
    log_p90_tracking_mean: float
    log_p90_tracking_std: float
    direct_label_names: list[str]
    constants: dict
    split_strategy: str
    split_seed: int
    train_group_count: int
    heldout_group_count: int
    train_row_count: int
    heldout_row_count: int
    original_row_count: int
    deduplicated_row_count: int
    removed_duplicate_rows: int
    sampler_config: dict

    # New frozen-benchmark metadata. The legacy heldout fields above remain aliases
    # for validation so older checkpoints and inference code still load.
    validation_group_count: int = 0
    test_group_count: int = 0
    validation_row_count: int = 0
    test_row_count: int = 0
    split_column: str = ""
    repeat_group_normalization: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BehaviorDataBundle:
    dataframe: pd.DataFrame
    train_dataframe: pd.DataFrame
    heldout_dataframe: pd.DataFrame
    train_dataset: "FlexDCBehaviorDataset"
    heldout_dataset: "FlexDCBehaviorDataset"
    train_loader: DataLoader
    train_eval_loader: DataLoader
    heldout_loader: DataLoader
    metadata: BehaviorDataMetadata
    audit: dict
    train_sampling_weights: np.ndarray | None

    # Heldout is retained as a backward-compatible alias for validation.
    test_dataframe: pd.DataFrame | None = None
    test_dataset: "FlexDCBehaviorDataset | None" = None
    test_loader: DataLoader | None = None

    @property
    def validation_dataframe(self) -> pd.DataFrame:
        return self.heldout_dataframe

    @property
    def validation_dataset(self) -> "FlexDCBehaviorDataset":
        return self.heldout_dataset

    @property
    def validation_loader(self) -> DataLoader:
        return self.heldout_loader


@dataclass
class TrainingResult:
    history: pd.DataFrame
    summary: dict
    checkpoint_paths: dict[str, str]


# ---------------------------------------------------------------------------
# Parsing and schema helpers
# ---------------------------------------------------------------------------

def choose_device(device_name: str = "auto") -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device option: {device_name}")


def parse_list(value, column_name: str) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if pd.isna(value):
        raise ValueError(f"Missing list value in {column_name}")
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        cleaned = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
        parsed = ast.literal_eval(cleaned)
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        raise ValueError(f"{column_name} must contain a list, got {type(parsed).__name__}")
    return [float(x) for x in parsed]


def parse_workload_mix(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(float, copy=True)
    else:
        text = str(value).strip()
        try:
            arr = np.asarray(json.loads(text), dtype=float)
        except Exception:
            arr = np.asarray(ast.literal_eval(text), dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (6, 7):
        raise ValueError(f"workload_mix must have shape [J,6] or [J,7], got {arr.shape}")
    return arr


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def read_results_and_diagnostics(
    results_csv: str | Path,
    diagnostics_csv: str | Path | None,
) -> pd.DataFrame:
    results = pd.read_csv(results_csv)
    if diagnostics_csv is None or str(diagnostics_csv).strip() == "":
        return results

    diagnostics = pd.read_csv(diagnostics_csv)
    keys = None
    for candidate in MERGE_KEYS_PRIORITY:
        if all(key in results.columns and key in diagnostics.columns for key in candidate):
            if not results.duplicated(candidate).any() and not diagnostics.duplicated(candidate).any():
                keys = candidate
                break
    if keys is None:
        raise ValueError(
            "Could not find a unique shared key for results/diagnostics merge. "
            "Expected Plan_Row_ID or Source_Output_Dir+Iteration."
        )

    extra_columns = [column for column in diagnostics.columns if column not in results.columns]
    merged = results.merge(
        diagnostics[keys + extra_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(results):
        raise ValueError(f"Merge changed row count: results={len(results)}, merged={len(merged)}")
    if extra_columns and merged[extra_columns].isna().all(axis=1).any():
        raise ValueError("At least one results row failed to match diagnostics.")
    return merged


def canonical_vector(values: Sequence[float], digits: int = 12) -> str:
    return json.dumps([round(float(x), digits) for x in values], separators=(",", ":"))


def _optional_seed_identity(df: pd.DataFrame) -> list[str]:
    for candidate in ["Simulation_Seed", "simulation_seed", "random_seed"]:
        if candidate in df.columns:
            return [candidate]
    return []


def _duplicate_identity_columns(df: pd.DataFrame) -> list[str]:
    required = [
        "Workload_Name",
        "server_count",
        "utilization",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "_weights_key",
    ]
    require_columns(df, required, "duplicate identity")
    return required[:-1] + _optional_seed_identity(df) + ["_weights_key"]


def deduplicate_behavior_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["_weights_key"] = df["weights"].apply(
        lambda x: canonical_vector(parse_list(x, "weights"))
    )
    identity = _duplicate_identity_columns(df)
    duplicate_mask = df.duplicated(identity, keep=False)

    check_columns = [
        "Mtrack_Error_MeanAbs_Normalized",
        "Ctrack_Epsilon_90th",
        "Simulator_RSR_Total_Cost",
        "Diagnostic_FullPaperObjective_Cost",
    ]
    disagreements = []
    if duplicate_mask.any():
        for _, group in df.loc[duplicate_mask].groupby(identity, dropna=False, sort=False):
            bad = False
            for column in check_columns:
                if column in group.columns and not np.allclose(
                    group[column].astype(float).to_numpy(),
                    float(group[column].astype(float).iloc[0]),
                    rtol=0,
                    atol=1e-10,
                ):
                    bad = True
            qos_keys = group["QoS_Delay_Probabilities"].apply(
                lambda x: canonical_vector(parse_list(x, "QoS_Delay_Probabilities"))
            )
            if qos_keys.nunique() != 1:
                bad = True
            if bad:
                disagreements.append(group[identity + check_columns].head(4).to_dict(orient="records"))
    if disagreements:
        raise ValueError(
            "Duplicate input configurations have inconsistent labels. "
            f"First examples: {disagreements[:2]}"
        )

    original_rows = len(df)
    df = df.drop_duplicates(identity, keep="first").reset_index(drop=True)
    return df, {
        "original_rows": int(original_rows),
        "deduplicated_rows": int(len(df)),
        "removed_duplicate_rows": int(original_rows - len(df)),
        "duplicate_rows_before_removal": int(duplicate_mask.sum()),
        "deduplication_identity": identity,
    }


def _preparse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_parsed_mix"] = df["workload_mix"].apply(parse_workload_mix)
    df["_parsed_weights"] = df["weights"].apply(lambda x: np.asarray(parse_list(x, "weights"), dtype=float))
    df["_parsed_qos"] = df["QoS_Delay_Probabilities"].apply(
        lambda x: np.asarray(parse_list(x, "QoS_Delay_Probabilities"), dtype=float)
    )
    df["_actual_max_pj"] = df["_parsed_qos"].apply(lambda x: float(np.max(x)))
    df["_actual_tracking_pass"] = df["Ctrack_Epsilon_90th"].astype(float) <= TRACKING_THRESHOLD
    df["_actual_qos_pass"] = df["_actual_max_pj"] <= QOS_THRESHOLD
    df["_actual_both_pass"] = df["_actual_tracking_pass"] & df["_actual_qos_pass"]
    df["_tracking_boundary"] = df["Ctrack_Epsilon_90th"].astype(float).between(0.20, 0.40)
    df["_qos_boundary"] = df["_parsed_qos"].apply(
        lambda x: bool(np.any((x >= 0.05) & (x <= 0.15)))
    )
    df["_context_id"] = (
        df["Workload_Name"].astype(str)
        + "|N=" + df["server_count"].astype(int).astype(str)
        + "|U=" + df["utilization"].astype(float).map(lambda x: f"{x:.6g}")
    )
    return df


def validate_behavior_dataframe(df: pd.DataFrame) -> dict:
    required = [
        "Workload_Name",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "Pbar_ratio",
        "R_ratio",
        "server_count",
        "utilization",
        "workload_mix_size",
        "workload_mix",
        "weights",
        "Mtrack_Error_MeanAbs_Normalized",
        "Ctrack_Epsilon_90th",
        "QoS_Delay_Probabilities",
        "Simulator_Power_Cost",
        "Simulator_RSR_Total_Cost",
        "Diagnostic_FullPaperObjective_Cost",
        "Mtrack_Price_Coefficient_piE",
        "Mtrack_Hour_Seconds",
        "R_actual_watts",
    ]
    require_columns(df, required, "FlexDC behavior training")

    j_values = []
    invalid_weights = []
    invalid_probabilities = []
    for index, row in df.iterrows():
        mix = row["_parsed_mix"] if "_parsed_mix" in row.index else parse_workload_mix(row["workload_mix"])
        weights = row["_parsed_weights"] if "_parsed_weights" in row.index else np.asarray(parse_list(row["weights"], "weights"), dtype=float)
        probabilities = row["_parsed_qos"] if "_parsed_qos" in row.index else np.asarray(parse_list(row["QoS_Delay_Probabilities"], "QoS_Delay_Probabilities"), dtype=float)
        j = int(mix.shape[0])
        j_values.append(j)
        if int(row["workload_mix_size"]) != j or len(weights) != j:
            invalid_weights.append(index)
        if len(probabilities) != j or not np.all(np.isfinite(probabilities)) or not np.all((probabilities >= 0) & (probabilities <= 1)):
            invalid_probabilities.append(index)
        if not np.isclose(weights.sum(), 1.0, atol=1e-8) or np.any(weights < 0):
            invalid_weights.append(index)
    if invalid_weights:
        raise ValueError(f"Invalid workload size/weights at rows: {sorted(set(invalid_weights))[:20]}")
    if invalid_probabilities:
        raise ValueError(f"Invalid QoS probability vectors at rows: {invalid_probabilities[:20]}")

    mean_tracking = df["Mtrack_Error_MeanAbs_Normalized"].astype(float).to_numpy()
    p90_tracking = df["Ctrack_Epsilon_90th"].astype(float).to_numpy()
    if not np.all(np.isfinite(mean_tracking)) or np.any(mean_tracking < 0):
        raise ValueError("Mean tracking labels must be finite and nonnegative.")
    if not np.all(np.isfinite(p90_tracking)) or np.any(p90_tracking < 0):
        raise ValueError("p90 tracking labels must be finite and nonnegative.")

    return {
        "rows": int(len(df)),
        "workload_sizes": sorted(int(x) for x in set(j_values)),
        "workloads": sorted(str(x) for x in df["Workload_Name"].unique()),
        "contexts": int(df["_context_id"].nunique()) if "_context_id" in df.columns else None,
        "server_counts": sorted(int(x) for x in df["server_count"].unique()),
        "utilizations": sorted(float(x) for x in df["utilization"].unique()),
        "actual_feasible_rows": int(df.get("_actual_both_pass", pd.Series(False, index=df.index)).sum()),
        "mean_tracking_min": float(mean_tracking.min()),
        "mean_tracking_max": float(mean_tracking.max()),
        "p90_tracking_min": float(p90_tracking.min()),
        "p90_tracking_max": float(p90_tracking.max()),
    }


# ---------------------------------------------------------------------------
# Engineered FlexDC features
# ---------------------------------------------------------------------------

def build_token_features_raw(
    mix: np.ndarray,
    weights: np.ndarray,
    *,
    utilization: float,
    server_count: int,
) -> np.ndarray:
    mix = np.asarray(mix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    j = int(len(weights))
    if mix.shape[0] != j:
        raise ValueError(f"Mix has {mix.shape[0]} rows but weights has {j} values")
    pmin_w, pmax_w, tmin, tmax, qos_threshold, job_size = [mix[:, i] for i in range(6)]
    runtime_ratio = tmax / np.maximum(tmin, 1e-9)
    qos_headroom = qos_threshold / np.maximum(runtime_ratio, 1e-9)
    queue_pressure = float(utilization) / np.maximum(j * weights, 1e-6)
    return np.column_stack(
        [
            pmin_w / 1000.0,
            pmax_w / 1000.0,
            (pmax_w - pmin_w) / 1000.0,
            np.log10(np.maximum(tmin, 1e-9)),
            np.log10(np.maximum(tmax, 1e-9)),
            runtime_ratio,
            qos_threshold,
            job_size,
            qos_headroom,
            weights,
            weights * j,
            weights * float(server_count) / 1000.0,
            queue_pressure,
        ]
    ).astype(np.float64)


def build_global_features_raw(row: pd.Series) -> np.ndarray:
    pbar = float(row["Pbar_kw_per_server"])
    reserve = float(row["R_kw_per_server"])
    server_count = int(row["server_count"])
    utilization = float(row["utilization"])
    j = int(row["workload_mix_size"])
    return np.asarray(
        [
            pbar,
            reserve,
            pbar - reserve,
            pbar + reserve,
            reserve / max(pbar, 1e-9),
            float(row["Pbar_ratio"]),
            float(row["R_ratio"]),
            utilization,
            server_count / 1000.0,
            math.log10(max(server_count, 1)),
            1.0 / max(j, 1),
            j / 8.0,
        ],
        dtype=np.float64,
    )


def _guarded_mean_std(values: np.ndarray, *, std_floor: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    bad = (~np.isfinite(std)) | (std < std_floor)
    std[bad] = 1.0
    return mean, std


def compute_feature_statistics(train_df: pd.DataFrame) -> dict:
    token_rows = []
    global_rows = []
    for _, row in train_df.iterrows():
        token_rows.append(
            build_token_features_raw(
                row["_parsed_mix"],
                row["_parsed_weights"],
                utilization=float(row["utilization"]),
                server_count=int(row["server_count"]),
            )
        )
        global_rows.append(build_global_features_raw(row))
    token_values = np.concatenate(token_rows, axis=0)
    global_values = np.stack(global_rows, axis=0)
    token_mean, token_std = _guarded_mean_std(token_values)
    global_mean, global_std = _guarded_mean_std(global_values)
    return {
        "token_mean": token_mean,
        "token_std": token_std,
        "global_mean": global_mean,
        "global_std": global_std,
    }


# ---------------------------------------------------------------------------
# Grouped split and balanced sampling
# ---------------------------------------------------------------------------

def _group_id_for_row(row: pd.Series) -> str:
    if "Base_Plan_Row_ID" in row.index and not pd.isna(row["Base_Plan_Row_ID"]):
        return str(row["Base_Plan_Row_ID"])
    pieces = [
        str(row["Workload_Name"]),
        str(int(row["server_count"])),
        f"{float(row['utilization']):.12g}",
        f"{float(row['Pbar_kw_per_server']):.12g}",
        f"{float(row['R_kw_per_server']):.12g}",
        canonical_vector(row["_parsed_weights"]),
    ]
    return hashlib.sha256("|".join(pieces).encode()).hexdigest()[:24]


def stratified_context_group_split(
    df: pd.DataFrame,
    *,
    heldout_fraction: float,
    split_seed: int,
) -> tuple[set[str], set[str], dict]:
    """Backward-compatible two-way split for older datasets.

    the frozen benchmark should use the preassigned Data_Split column instead.
    """
    rng = np.random.default_rng(split_seed)
    train_groups: set[str] = set()
    heldout_groups: set[str] = set()
    context_rows = []
    for context_id, context_df in df.groupby("_context_id", sort=True):
        groups = np.asarray(sorted(context_df["_group_id"].unique()), dtype=object)
        rng.shuffle(groups)
        if len(groups) <= 1:
            n_heldout = 0
        else:
            n_heldout = int(round(len(groups) * heldout_fraction))
            n_heldout = min(max(n_heldout, 1), len(groups) - 1)
        held = set(str(x) for x in groups[:n_heldout])
        train = set(str(x) for x in groups[n_heldout:])
        train_groups.update(train)
        heldout_groups.update(held)
        context_rows.append(
            {
                "context_id": context_id,
                "groups": int(len(groups)),
                "train_groups": int(len(train)),
                "validation_groups": int(len(held)),
                "test_groups": 0,
                "rows": int(len(context_df)),
            }
        )
    if train_groups & heldout_groups:
        raise AssertionError("Grouped split leakage detected.")
    return train_groups, heldout_groups, {"context_split_summary": context_rows}


def preassigned_train_validation_test_split(
    df: pd.DataFrame,
    *,
    split_column: str = "Data_Split",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Use the split frozen by the the frozen benchmark plan generator.

    Every Base_Plan_Row_ID must belong to exactly one split, which keeps all
    seed realizations of the same operating configuration together.
    """
    if split_column not in df.columns:
        raise ValueError(
            f"Preassigned split requested, but {split_column!r} is missing."
        )

    normalized = df[split_column].astype(str).str.strip().str.lower()
    aliases = {
        "train": "train",
        "training": "train",
        "validation": "validation",
        "valid": "validation",
        "val": "validation",
        "heldout": "validation",
        "test": "test",
    }
    unknown = sorted(set(normalized) - set(aliases))
    if unknown:
        raise ValueError(
            f"Unknown values in {split_column}: {unknown}. "
            "Expected train, validation, or test."
        )
    normalized = normalized.map(aliases)
    df = df.copy()
    df["_data_split"] = normalized

    required = {"train", "validation", "test"}
    present = set(df["_data_split"].unique())
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"Preassigned dataset is missing split(s): {missing}")

    group_split_counts = df.groupby("_group_id")["_data_split"].nunique()
    crossing = group_split_counts[group_split_counts != 1]
    if len(crossing):
        raise AssertionError(
            f"{len(crossing)} base configuration groups cross splits; "
            f"examples={crossing.index[:10].tolist()}"
        )

    train_df = df[df["_data_split"] == "train"].copy().reset_index(drop=True)
    validation_df = df[df["_data_split"] == "validation"].copy().reset_index(drop=True)
    test_df = df[df["_data_split"] == "test"].copy().reset_index(drop=True)

    split_sets = {
        name: set(part["_group_id"].astype(str))
        for name, part in [
            ("train", train_df),
            ("validation", validation_df),
            ("test", test_df),
        ]
    }
    if split_sets["train"] & split_sets["validation"]:
        raise AssertionError("Train/validation group leakage detected.")
    if split_sets["train"] & split_sets["test"]:
        raise AssertionError("Train/test group leakage detected.")
    if split_sets["validation"] & split_sets["test"]:
        raise AssertionError("Validation/test group leakage detected.")

    context_rows = []
    for context_id, context_df in df.groupby("_context_id", sort=True):
        record = {
            "context_id": context_id,
            "groups": int(context_df["_group_id"].nunique()),
            "rows": int(len(context_df)),
        }
        for split_name in ["train", "validation", "test"]:
            part = context_df[context_df["_data_split"] == split_name]
            record[f"{split_name}_groups"] = int(part["_group_id"].nunique())
            record[f"{split_name}_rows"] = int(len(part))
        context_rows.append(record)

    audit = {
        "context_split_summary": context_rows,
        "split_column": split_column,
        "split_rows": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
        "split_groups": {
            name: int(len(groups)) for name, groups in split_sets.items()
        },
        "group_overlap": 0,
    }
    return train_df, validation_df, test_df, audit


def build_training_sampling_weights(
    train_df: pd.DataFrame,
    *,
    mode: str = "balanced",
    natural_fraction: float = 0.50,
    context_fraction: float = 0.25,
    priority_fraction: float = 0.25,
    feasible_boost: float = 4.0,
    tracking_boundary_boost: float = 2.0,
    qos_boundary_boost: float = 3.0,
    repeat_group_normalization: bool = True,
) -> tuple[np.ndarray | None, dict]:
    """Build row probabilities while treating seed repeats as one base input.

    the frozen benchmark repeats five percent of base configurations with extra seeds.
    Without normalization, a three-seed configuration would receive three
    times the training probability of an otherwise identical one-seed
    configuration.  We first assign probability to Base_Plan_Row_ID groups
    and then divide that probability equally among each group's seed rows.
    """
    mode = str(mode).lower().strip()
    if mode not in {"balanced", "natural", "shuffle", "none"}:
        raise ValueError(f"Unknown sampler mode: {mode}")

    if "_group_id" not in train_df.columns:
        raise ValueError("Training dataframe is missing _group_id.")

    fractions = np.asarray(
        [natural_fraction, context_fraction, priority_fraction],
        dtype=float,
    )
    if mode == "balanced":
        if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-8):
            raise ValueError(
                "natural/context/priority sampler fractions must be "
                "nonnegative and sum to 1"
            )
    else:
        fractions = np.asarray([1.0, 0.0, 0.0], dtype=float)

    working = train_df.copy()
    group_sizes = working.groupby("_group_id")["_group_id"].transform("size").astype(float)
    if repeat_group_normalization:
        row_share_within_group = 1.0 / group_sizes.to_numpy()
    else:
        row_share_within_group = np.ones(len(working), dtype=float)

    group_table = (
        working.groupby("_group_id", sort=False)
        .agg(
            _context_id=("_context_id", "first"),
            group_rows=("_group_id", "size"),
            feasible_fraction=("_actual_both_pass", "mean"),
            tracking_boundary_fraction=("_tracking_boundary", "mean"),
            qos_boundary_fraction=("_qos_boundary", "mean"),
        )
        .reset_index()
    )
    group_ids = group_table["_group_id"].astype(str).tolist()
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    row_group_index = np.asarray(
        [group_index[str(value)] for value in working["_group_id"]],
        dtype=int,
    )

    num_groups = max(len(group_table), 1)
    natural_group = np.full(num_groups, 1.0 / num_groups, dtype=float)

    context_group_counts = group_table["_context_id"].value_counts()
    num_contexts = max(len(context_group_counts), 1)
    context_group = np.asarray(
        [
            1.0 / (num_contexts * context_group_counts[context_id])
            for context_id in group_table["_context_id"]
        ],
        dtype=float,
    )

    priority_group_score = np.ones(num_groups, dtype=float)
    priority_group_score *= 1.0 + (feasible_boost - 1.0) * group_table[
        "feasible_fraction"
    ].to_numpy(float)
    priority_group_score *= 1.0 + (tracking_boundary_boost - 1.0) * group_table[
        "tracking_boundary_fraction"
    ].to_numpy(float)
    priority_group_score *= 1.0 + (qos_boundary_boost - 1.0) * group_table[
        "qos_boundary_fraction"
    ].to_numpy(float)
    priority_group = priority_group_score / priority_group_score.sum()

    group_probabilities = (
        fractions[0] * natural_group
        + fractions[1] * context_group
        + fractions[2] * priority_group
    )
    group_probabilities = group_probabilities / group_probabilities.sum()

    if repeat_group_normalization:
        probabilities = group_probabilities[row_group_index] * row_share_within_group
    else:
        probabilities = group_probabilities[row_group_index]
    probabilities = probabilities / probabilities.sum()

    actual_group_totals = pd.Series(probabilities).groupby(working["_group_id"].astype(str)).sum()
    target_group_totals = pd.Series(
        group_probabilities,
        index=pd.Index(group_ids, name="_group_id"),
    )
    aligned = actual_group_totals.reindex(target_group_totals.index)
    max_group_probability_error = float(
        np.max(np.abs(aligned.to_numpy() - target_group_totals.to_numpy()))
    )

    expected_feasible_share = float(
        probabilities[working["_actual_both_pass"].to_numpy(bool)].sum()
    )
    expected_tracking_boundary_share = float(
        probabilities[working["_tracking_boundary"].to_numpy(bool)].sum()
    )
    expected_qos_boundary_share = float(
        probabilities[working["_qos_boundary"].to_numpy(bool)].sum()
    )

    audit = {
        "mode": "balanced" if mode == "balanced" else "natural_base_groups",
        "repeat_group_normalization": bool(repeat_group_normalization),
        "base_groups": int(num_groups),
        "repeated_base_groups": int((group_table["group_rows"] > 1).sum()),
        "maximum_rows_in_base_group": int(group_table["group_rows"].max()),
        "max_group_probability_error": max_group_probability_error,
        "natural_fraction": float(fractions[0]),
        "context_fraction": float(fractions[1]),
        "priority_fraction": float(fractions[2]),
        "feasible_boost": float(feasible_boost),
        "tracking_boundary_boost": float(tracking_boundary_boost),
        "qos_boundary_boost": float(qos_boundary_boost),
        "raw_feasible_share": float(working["_actual_both_pass"].mean()),
        "expected_sampled_feasible_share": expected_feasible_share,
        "raw_tracking_boundary_share": float(working["_tracking_boundary"].mean()),
        "expected_sampled_tracking_boundary_share": expected_tracking_boundary_share,
        "raw_qos_boundary_share": float(working["_qos_boundary"].mean()),
        "expected_sampled_qos_boundary_share": expected_qos_boundary_share,
    }
    return probabilities, audit


class EpochWeightedRandomSampler(Sampler[int]):
    """Deterministic weighted sampler whose draw order is keyed by epoch.

    This makes a resumed run use the same sampling distribution/order it would
    have used without interruption for that epoch.
    """

    def __init__(self, weights: np.ndarray, *, num_samples: int, base_seed: int) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class FlexDCBehaviorDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        token_mean: np.ndarray,
        token_std: np.ndarray,
        global_mean: np.ndarray,
        global_std: np.ndarray,
        constants: FlexDCBehaviorConstants | None = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.token_mean = np.asarray(token_mean, dtype=np.float64)
        self.token_std = np.asarray(token_std, dtype=np.float64)
        self.global_mean = np.asarray(global_mean, dtype=np.float64)
        self.global_std = np.asarray(global_std, dtype=np.float64)
        self.constants = constants or FlexDCBehaviorConstants()

        self.features = []
        self.workloads = []
        self.qos_targets = []
        for _, row in self.df.iterrows():
            raw_global = build_global_features_raw(row)
            raw_tokens = build_token_features_raw(
                row["_parsed_mix"],
                row["_parsed_weights"],
                utilization=float(row["utilization"]),
                server_count=int(row["server_count"]),
            )
            self.features.append(
                torch.tensor((raw_global - self.global_mean) / self.global_std, dtype=torch.float32)
            )
            self.workloads.append(
                torch.tensor((raw_tokens - self.token_mean) / self.token_std, dtype=torch.float32)
            )
            self.qos_targets.append(torch.tensor(row["_parsed_qos"], dtype=torch.float32))
        self.features = torch.stack(self.features)

        mean_tracking = self.df["Mtrack_Error_MeanAbs_Normalized"].astype(float).to_numpy()
        p90_tracking = self.df["Ctrack_Epsilon_90th"].astype(float).to_numpy()
        self.log_tracking = torch.tensor(
            np.column_stack(
                [
                    np.log(mean_tracking + self.constants.mean_tracking_log_floor),
                    np.log(p90_tracking + self.constants.p90_tracking_log_floor),
                ]
            ),
            dtype=torch.float32,
        )
        self.raw_mean_tracking = torch.tensor(mean_tracking, dtype=torch.float32)
        self.raw_p90_tracking = torch.tensor(p90_tracking, dtype=torch.float32)
        self.simulator_power_cost = torch.tensor(self.df["Simulator_Power_Cost"].astype(float).to_numpy(), dtype=torch.float32)
        self.actual_m_rsr = torch.tensor(self.df["Simulator_RSR_Total_Cost"].astype(float).to_numpy(), dtype=torch.float32)
        self.actual_objective = torch.tensor(self.df["Diagnostic_FullPaperObjective_Cost"].astype(float).to_numpy(), dtype=torch.float32)
        self.r_actual_watts = torch.tensor(self.df["R_actual_watts"].astype(float).to_numpy(), dtype=torch.float32)
        self.tracking_price_coefficient = torch.tensor(self.df["Mtrack_Price_Coefficient_piE"].astype(float).to_numpy(), dtype=torch.float32)
        self.hour_seconds = torch.tensor(self.df["Mtrack_Hour_Seconds"].astype(float).to_numpy(), dtype=torch.float32)
        self.plan_row_ids = (
            self.df["Plan_Row_ID"].astype(str).tolist()
            if "Plan_Row_ID" in self.df.columns
            else [str(index) for index in self.df.index]
        )
        self.workload_names = self.df["Workload_Name"].astype(str).tolist()
        self.context_ids = self.df["_context_id"].astype(str).tolist()
        self.server_counts = self.df["server_count"].astype(int).tolist()
        self.utilizations = self.df["utilization"].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        return {
            "features": self.features[index],
            "workload": self.workloads[index],
            "tracking_logs": self.log_tracking[index],
            "qos_probabilities": self.qos_targets[index],
            "raw_mean_tracking": self.raw_mean_tracking[index],
            "raw_p90_tracking": self.raw_p90_tracking[index],
            "simulator_power_cost": self.simulator_power_cost[index],
            "actual_m_rsr": self.actual_m_rsr[index],
            "actual_objective": self.actual_objective[index],
            "r_actual_watts": self.r_actual_watts[index],
            "tracking_price_coefficient": self.tracking_price_coefficient[index],
            "hour_seconds": self.hour_seconds[index],
            "plan_row_id": self.plan_row_ids[index],
            "workload_name": self.workload_names[index],
            "context_id": self.context_ids[index],
            "server_count": self.server_counts[index],
            "utilization": self.utilizations[index],
        }


def collate_behavior_batch(items: list[dict]) -> dict:
    workloads = [item["workload"] for item in items]
    qos_targets = [item["qos_probabilities"] for item in items]
    lengths = torch.tensor([workload.size(0) for workload in workloads], dtype=torch.long)
    padded_workloads = pad_sequence(workloads, batch_first=True, padding_value=0.0)
    padded_qos = pad_sequence(qos_targets, batch_first=True, padding_value=0.0)
    positions = torch.arange(padded_workloads.size(1)).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)

    def stack(key: str) -> torch.Tensor:
        return torch.stack([item[key] for item in items])

    return {
        "features": stack("features"),
        "workload": padded_workloads,
        "mask": mask,
        "tracking_logs": stack("tracking_logs"),
        "qos_probabilities": padded_qos,
        "raw_mean_tracking": stack("raw_mean_tracking"),
        "raw_p90_tracking": stack("raw_p90_tracking"),
        "simulator_power_cost": stack("simulator_power_cost"),
        "actual_m_rsr": stack("actual_m_rsr"),
        "actual_objective": stack("actual_objective"),
        "r_actual_watts": stack("r_actual_watts"),
        "tracking_price_coefficient": stack("tracking_price_coefficient"),
        "hour_seconds": stack("hour_seconds"),
        "plan_row_id": [item["plan_row_id"] for item in items],
        "workload_name": [item["workload_name"] for item in items],
        "context_id": [item["context_id"] for item in items],
        "server_count": [item["server_count"] for item in items],
        "utilization": [item["utilization"] for item in items],
    }


def prepare_behavior_data(
    *,
    results_csv: str | Path,
    diagnostics_csv: str | Path | None,
    batch_size: int = 2048,
    heldout_fraction: float = 0.30,
    split_seed: int = 0,
    split_mode: str = "auto",
    split_column: str = "Data_Split",
    num_workers: int = 0,
    deduplicate: bool = False,
    constants: FlexDCBehaviorConstants | None = None,
    sampler_mode: str = "balanced",
    sampler_seed: int = 0,
    natural_fraction: float = 0.50,
    context_fraction: float = 0.25,
    priority_fraction: float = 0.25,
    feasible_boost: float = 4.0,
    tracking_boundary_boost: float = 2.0,
    qos_boundary_boost: float = 3.0,
    repeat_group_normalization: bool = True,
) -> BehaviorDataBundle:
    """Prepare behavior-model data with frozen-benchmark split support.

    ``split_mode='auto'`` uses the preassigned train/validation/test split when
    ``Data_Split`` is present. Older datasets without that column retain the
    previous grouped 70/30 train/validation behavior and have no test set.
    """
    constants = constants or FlexDCBehaviorConstants()
    merged = read_results_and_diagnostics(results_csv, diagnostics_csv)
    original_rows = len(merged)

    if deduplicate:
        merged, duplicate_audit = deduplicate_behavior_rows(merged)
    else:
        duplicate_audit = {
            "original_rows": int(original_rows),
            "deduplicated_rows": int(original_rows),
            "removed_duplicate_rows": 0,
            "duplicate_rows_before_removal": 0,
            "deduplication_identity": None,
        }

    merged = _preparse_dataframe(merged)
    schema_audit = validate_behavior_dataframe(merged)
    merged["_group_id"] = merged.apply(_group_id_for_row, axis=1)

    group_sizes = merged.groupby("_group_id")["_group_id"].transform("size").astype(int)
    computed_group_weight = 1.0 / group_sizes.astype(float)
    merged["_base_group_row_count"] = group_sizes
    merged["_base_group_normalization_weight"] = computed_group_weight

    supplied_group_weight_error = 0.0
    if "Base_Group_Normalization_Weight" in merged.columns:
        supplied = pd.to_numeric(
            merged["Base_Group_Normalization_Weight"], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(supplied).all():
            raise ValueError("Base_Group_Normalization_Weight contains non-finite values.")
        supplied_group_weight_error = float(
            np.max(np.abs(supplied - computed_group_weight.to_numpy(float)))
        )
        if supplied_group_weight_error > 1e-10:
            raise ValueError(
                "Base_Group_Normalization_Weight disagrees with Base_Plan_Row_ID "
                f"group sizes; max error={supplied_group_weight_error}."
            )

    split_mode_normalized = str(split_mode).strip().lower()
    if split_mode_normalized not in {"auto", "preassigned", "generated", "legacy"}:
        raise ValueError(
            "split_mode must be auto, preassigned, generated, or legacy"
        )
    use_preassigned = (
        split_mode_normalized == "preassigned"
        or (split_mode_normalized == "auto" and split_column in merged.columns)
    )

    if use_preassigned:
        train_df, validation_df, test_df, split_audit = (
            preassigned_train_validation_test_split(
                merged, split_column=split_column
            )
        )
        split_strategy = f"preassigned_{split_column}_grouped_BasePlan"
        train_groups = set(train_df["_group_id"].astype(str))
        validation_groups = set(validation_df["_group_id"].astype(str))
        test_groups = set(test_df["_group_id"].astype(str))
    else:
        train_groups, validation_groups, split_audit = stratified_context_group_split(
            merged,
            heldout_fraction=heldout_fraction,
            split_seed=split_seed,
        )
        test_groups: set[str] = set()
        train_df = merged[merged["_group_id"].isin(train_groups)].copy().reset_index(drop=True)
        validation_df = merged[merged["_group_id"].isin(validation_groups)].copy().reset_index(drop=True)
        test_df = merged.iloc[0:0].copy().reset_index(drop=True)
        split_strategy = "context_stratified_grouped_BasePlan_or_exact_input"

    if train_groups & validation_groups:
        raise AssertionError("Train/validation group leakage detected.")
    if train_groups & test_groups:
        raise AssertionError("Train/test group leakage detected.")
    if validation_groups & test_groups:
        raise AssertionError("Validation/test group leakage detected.")
    if len(train_df) == 0 or len(validation_df) == 0:
        raise ValueError("Training and validation splits must both be non-empty.")
    if use_preassigned and len(test_df) == 0:
        raise ValueError("Preassigned frozen-benchmark split must include a non-empty test set.")

    # No validation or test information contributes to normalization.
    feature_stats = compute_feature_statistics(train_df)
    train_mean = train_df["Mtrack_Error_MeanAbs_Normalized"].astype(float).to_numpy()
    train_p90 = train_df["Ctrack_Epsilon_90th"].astype(float).to_numpy()
    log_mean = np.log(train_mean + constants.mean_tracking_log_floor)
    log_p90 = np.log(train_p90 + constants.p90_tracking_log_floor)
    log_mean_std = float(np.std(log_mean))
    log_p90_std = float(np.std(log_p90))
    if not np.isfinite(log_mean_std) or log_mean_std < 1e-8:
        log_mean_std = 1.0
    if not np.isfinite(log_p90_std) or log_p90_std < 1e-8:
        log_p90_std = 1.0

    def make_dataset(frame: pd.DataFrame) -> FlexDCBehaviorDataset:
        return FlexDCBehaviorDataset(
            frame,
            token_mean=feature_stats["token_mean"],
            token_std=feature_stats["token_std"],
            global_mean=feature_stats["global_mean"],
            global_std=feature_stats["global_std"],
            constants=constants,
        )

    train_dataset = make_dataset(train_df)
    validation_dataset = make_dataset(validation_df)
    test_dataset = make_dataset(test_df) if len(test_df) else None

    sample_weights, sampler_audit = build_training_sampling_weights(
        train_df,
        mode=sampler_mode,
        natural_fraction=natural_fraction,
        context_fraction=context_fraction,
        priority_fraction=priority_fraction,
        feasible_boost=feasible_boost,
        tracking_boundary_boost=tracking_boundary_boost,
        qos_boundary_boost=qos_boundary_boost,
        repeat_group_normalization=repeat_group_normalization,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_behavior_batch,
        pin_memory=torch.cuda.is_available(),
    )
    if sample_weights is None:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    else:
        sampler = EpochWeightedRandomSampler(
            sample_weights,
            num_samples=len(train_dataset),
            base_seed=int(sampler_seed),
        )
        train_loader = DataLoader(
            train_dataset, sampler=sampler, shuffle=False, **loader_kwargs
        )
    train_eval_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs
    )
    test_loader = (
        DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        if test_dataset is not None
        else None
    )

    metadata = BehaviorDataMetadata(
        token_feature_names=list(TOKEN_FEATURE_NAMES),
        token_feature_mean=feature_stats["token_mean"].tolist(),
        token_feature_std=feature_stats["token_std"].tolist(),
        global_feature_names=list(GLOBAL_FEATURE_NAMES),
        global_feature_mean=feature_stats["global_mean"].tolist(),
        global_feature_std=feature_stats["global_std"].tolist(),
        log_mean_tracking_mean=float(np.mean(log_mean)),
        log_mean_tracking_std=log_mean_std,
        log_p90_tracking_mean=float(np.mean(log_p90)),
        log_p90_tracking_std=log_p90_std,
        direct_label_names=[
            "log_mean_tracking",
            "log_p90_tracking",
            "P_j_per_real_job_type",
        ],
        constants=constants.to_dict(),
        split_strategy=split_strategy,
        split_seed=int(split_seed),
        train_group_count=int(len(train_groups)),
        heldout_group_count=int(len(validation_groups)),
        train_row_count=int(len(train_df)),
        heldout_row_count=int(len(validation_df)),
        original_row_count=int(original_rows),
        deduplicated_row_count=int(len(merged)),
        removed_duplicate_rows=int(duplicate_audit["removed_duplicate_rows"]),
        sampler_config=sampler_audit,
        validation_group_count=int(len(validation_groups)),
        test_group_count=int(len(test_groups)),
        validation_row_count=int(len(validation_df)),
        test_row_count=int(len(test_df)),
        split_column=split_column if use_preassigned else "",
        repeat_group_normalization=bool(repeat_group_normalization),
    )

    audit = {
        **duplicate_audit,
        **schema_audit,
        **split_audit,
        "split_strategy": split_strategy,
        "used_preassigned_split": bool(use_preassigned),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(len(train_groups)),
        "validation_groups": int(len(validation_groups)),
        "test_groups": int(len(test_groups)),
        "group_overlap": 0,
        "train_actual_feasible_rows": int(train_df["_actual_both_pass"].sum()),
        "validation_actual_feasible_rows": int(validation_df["_actual_both_pass"].sum()),
        "test_actual_feasible_rows": int(test_df["_actual_both_pass"].sum()),
        "repeated_base_groups": int(
            (merged.groupby("_group_id").size() > 1).sum()
        ),
        "base_group_normalization_max_input_error": supplied_group_weight_error,
        "sampler": sampler_audit,
    }

    return BehaviorDataBundle(
        dataframe=merged,
        train_dataframe=train_df,
        heldout_dataframe=validation_df,
        train_dataset=train_dataset,
        heldout_dataset=validation_dataset,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        heldout_loader=validation_loader,
        metadata=metadata,
        audit=audit,
        train_sampling_weights=sample_weights,
        test_dataframe=test_df,
        test_dataset=test_dataset,
        test_loader=test_loader,
    )


# ---------------------------------------------------------------------------
# Loss, reconstruction, metrics
# ---------------------------------------------------------------------------

def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def decode_tracking_logs(
    logs: torch.Tensor,
    constants: FlexDCBehaviorConstants,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_tracking = torch.clamp(
        torch.exp(logs[:, 0]) - constants.mean_tracking_log_floor,
        min=0.0,
    )
    p90_tracking = torch.clamp(
        torch.exp(logs[:, 1]) - constants.p90_tracking_log_floor,
        min=0.0,
    )
    return mean_tracking, p90_tracking


def reconstruct_flexdc_outputs(
    *,
    tracking_logs: torch.Tensor,
    qos_probabilities: torch.Tensor,
    mask: torch.Tensor,
    simulator_power_cost: torch.Tensor,
    r_actual_watts: torch.Tensor,
    tracking_price_coefficient: torch.Tensor,
    hour_seconds: torch.Tensor,
    constants: FlexDCBehaviorConstants,
) -> dict[str, torch.Tensor]:
    mean_tracking, p90_tracking = decode_tracking_logs(tracking_logs, constants)
    m_track = r_actual_watts * mean_tracking * tracking_price_coefficient * hour_seconds
    m_rsr = simulator_power_cost + m_track
    ctrack = constants.ctrack_psi * F.softplus(
        constants.ctrack_mu * (p90_tracking - constants.tracking_threshold)
    )
    per_job_cqos = constants.qos_beta * F.softplus(
        constants.qos_rho * (qos_probabilities - constants.qos_threshold)
    )
    cqos = (per_job_cqos * mask.to(per_job_cqos.dtype)).sum(dim=1)
    objective = m_rsr + ctrack + cqos
    return {
        "mean_tracking": mean_tracking,
        "p90_tracking": p90_tracking,
        "m_track": m_track,
        "m_rsr": m_rsr,
        "ctrack": ctrack,
        "cqos": cqos,
        "objective": objective,
    }


def behavior_loss(
    model_output: dict[str, torch.Tensor],
    batch: dict,
    metadata: BehaviorDataMetadata,
    *,
    mean_tracking_weight: float = 1.0,
    p90_tracking_weight: float = 1.0,
    qos_weight: float = 4.0,
    tracking_boundary_multiplier: float = 2.0,
    qos_boundary_multiplier: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_logs = model_output["tracking_logs"]
    true_logs = batch["tracking_logs"]
    mean_std = max(float(metadata.log_mean_tracking_std), 1e-12)
    p90_std = max(float(metadata.log_p90_tracking_std), 1e-12)

    mean_per_row = F.smooth_l1_loss(
        predicted_logs[:, 0] / mean_std,
        true_logs[:, 0] / mean_std,
        reduction="none",
    )
    mean_loss = mean_per_row.mean()

    p90_per_row = F.smooth_l1_loss(
        predicted_logs[:, 1] / p90_std,
        true_logs[:, 1] / p90_std,
        reduction="none",
    )
    p90_boundary = ((batch["raw_p90_tracking"] >= 0.20) & (batch["raw_p90_tracking"] <= 0.40)).to(p90_per_row.dtype)
    p90_row_weights = 1.0 + (tracking_boundary_multiplier - 1.0) * p90_boundary
    p90_loss = (p90_per_row * p90_row_weights).sum() / p90_row_weights.sum().clamp_min(1.0)

    mask = batch["mask"].to(predicted_logs.dtype)
    squared_qos_error = (
        model_output["qos_probabilities"] - batch["qos_probabilities"]
    ) ** 2
    qos_boundary = (
        (batch["qos_probabilities"] >= 0.05)
        & (batch["qos_probabilities"] <= 0.15)
        & batch["mask"]
    ).to(squared_qos_error.dtype)
    qos_element_weights = mask * (
        1.0 + (qos_boundary_multiplier - 1.0) * qos_boundary
    )
    qos_loss = (squared_qos_error * qos_element_weights).sum() / qos_element_weights.sum().clamp_min(1.0)

    total = (
        mean_tracking_weight * mean_loss
        + p90_tracking_weight * p90_loss
        + qos_weight * qos_loss
    )
    return total, {
        "loss_total": total,
        "loss_log_mean_tracking_huber": mean_loss,
        "loss_log_p90_tracking_huber": p90_loss,
        "loss_qos_brier": qos_loss,
    }


def _safe_r2(true: np.ndarray, pred: np.ndarray) -> float:
    if len(true) < 2 or np.allclose(true, true[0]):
        return float("nan")
    return float(r2_score(true, pred))


def _regression_metrics(true: np.ndarray, pred: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}/mae": float(mean_absolute_error(true, pred)),
        f"{prefix}/rmse": float(math.sqrt(mean_squared_error(true, pred))),
        f"{prefix}/r2": _safe_r2(true, pred),
        f"{prefix}/actual_mean": float(np.mean(true)),
        f"{prefix}/pred_mean": float(np.mean(pred)),
        f"{prefix}/bias_pred_minus_actual": float(np.mean(pred - true)),
    }


def _classification_metrics(actual: np.ndarray, predicted: np.ndarray, prefix: str) -> dict:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    total = len(actual)
    actual_feasible = tp + fn
    actual_infeasible = tn + fp
    predicted_feasible = tp + fp
    precision = tp / predicted_feasible if predicted_feasible else float("nan")
    recall = tp / actual_feasible if actual_feasible else float("nan")
    specificity = tn / actual_infeasible if actual_infeasible else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan")
    balanced = np.nanmean([recall, specificity])
    return {
        f"{prefix}/accuracy_overall": (tp + tn) / total if total else float("nan"),
        f"{prefix}/actual_feasible_accuracy": recall,
        f"{prefix}/actual_infeasible_accuracy": specificity,
        f"{prefix}/feasible_precision": precision,
        f"{prefix}/f1_feasible": f1,
        f"{prefix}/balanced_accuracy": float(balanced),
        f"{prefix}/false_feasible_rate": fp / actual_infeasible if actual_infeasible else float("nan"),
        f"{prefix}/false_infeasible_rate": fn / actual_feasible if actual_feasible else float("nan"),
        f"{prefix}/true_feasible_count": float(tp),
        f"{prefix}/true_infeasible_count": float(tn),
        f"{prefix}/false_feasible_count": float(fp),
        f"{prefix}/false_infeasible_count": float(fn),
        f"{prefix}/actual_feasible_count": float(actual_feasible),
        f"{prefix}/actual_infeasible_count": float(actual_infeasible),
        f"{prefix}/predicted_feasible_count": float(predicted_feasible),
    }


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def _subset_summary_metrics(
    *,
    true_p90: np.ndarray,
    pred_p90: np.ndarray,
    true_max_qos: np.ndarray,
    pred_max_qos: np.ndarray,
    mask: np.ndarray,
    prefix: str,
    constants: FlexDCBehaviorConstants,
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {}
    actual = (true_p90[mask] <= constants.tracking_threshold) & (
        true_max_qos[mask] <= constants.qos_threshold
    )
    predicted = (pred_p90[mask] <= constants.tracking_threshold) & (
        pred_max_qos[mask] <= constants.qos_threshold
    )
    metrics = {
        f"{prefix}/p90_mae": float(np.mean(np.abs(pred_p90[mask] - true_p90[mask]))),
        f"{prefix}/max_pj_mae": float(np.mean(np.abs(pred_max_qos[mask] - true_max_qos[mask]))),
    }
    metrics.update(_classification_metrics(actual, predicted, f"{prefix}/feasibility"))
    return metrics


def evaluate_behavior_loader(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    metadata: BehaviorDataMetadata,
    prefix: str,
    mean_tracking_weight: float = 1.0,
    p90_tracking_weight: float = 1.0,
    qos_weight: float = 4.0,
    tracking_boundary_multiplier: float = 2.0,
    qos_boundary_multiplier: float = 2.0,
    return_rows: bool = False,
    include_workload_metrics: bool = False,
) -> tuple[dict, pd.DataFrame | None]:
    model.eval()
    constants = FlexDCBehaviorConstants(**metadata.constants)
    accum: dict[str, list] = {
        "true_logs": [], "pred_logs": [],
        "true_mean": [], "pred_mean": [],
        "true_p90": [], "pred_p90": [],
        "true_qos": [], "pred_qos": [],
        "true_max_qos": [], "pred_max_qos": [],
        "true_m_rsr": [], "pred_m_rsr": [],
        "true_objective": [], "pred_objective": [],
        "plan_row_ids": [], "workload_names": [], "context_ids": [],
        "server_counts": [], "utilizations": [],
    }
    loss_sums = {
        "loss_total": 0.0,
        "loss_log_mean_tracking_huber": 0.0,
        "loss_log_p90_tracking_huber": 0.0,
        "loss_qos_brier": 0.0,
    }
    row_count = 0

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(batch["features"], batch["workload"], batch["mask"])
            _, losses = behavior_loss(
                output,
                batch,
                metadata,
                mean_tracking_weight=mean_tracking_weight,
                p90_tracking_weight=p90_tracking_weight,
                qos_weight=qos_weight,
                tracking_boundary_multiplier=tracking_boundary_multiplier,
                qos_boundary_multiplier=qos_boundary_multiplier,
            )
            batch_size = batch["features"].size(0)
            row_count += batch_size
            for name, value in losses.items():
                loss_sums[name] += float(value.detach().cpu()) * batch_size

            reconstructed = reconstruct_flexdc_outputs(
                tracking_logs=output["tracking_logs"],
                qos_probabilities=output["qos_probabilities"],
                mask=batch["mask"],
                simulator_power_cost=batch["simulator_power_cost"],
                r_actual_watts=batch["r_actual_watts"],
                tracking_price_coefficient=batch["tracking_price_coefficient"],
                hour_seconds=batch["hour_seconds"],
                constants=constants,
            )
            real_mask = batch["mask"].detach().cpu().numpy().astype(bool)
            true_qos = batch["qos_probabilities"].detach().cpu().numpy()
            pred_qos = output["qos_probabilities"].detach().cpu().numpy()
            true_masked = np.where(real_mask, true_qos, -np.inf)
            pred_masked = np.where(real_mask, pred_qos, -np.inf)

            accum["true_logs"].append(batch["tracking_logs"].detach().cpu().numpy())
            accum["pred_logs"].append(output["tracking_logs"].detach().cpu().numpy())
            accum["true_mean"].append(batch["raw_mean_tracking"].detach().cpu().numpy())
            accum["pred_mean"].append(reconstructed["mean_tracking"].detach().cpu().numpy())
            accum["true_p90"].append(batch["raw_p90_tracking"].detach().cpu().numpy())
            accum["pred_p90"].append(reconstructed["p90_tracking"].detach().cpu().numpy())
            accum["true_qos"].append(true_qos[real_mask])
            accum["pred_qos"].append(pred_qos[real_mask])
            accum["true_max_qos"].append(np.max(true_masked, axis=1))
            accum["pred_max_qos"].append(np.max(pred_masked, axis=1))
            accum["true_m_rsr"].append(batch["actual_m_rsr"].detach().cpu().numpy())
            accum["pred_m_rsr"].append(reconstructed["m_rsr"].detach().cpu().numpy())
            accum["true_objective"].append(batch["actual_objective"].detach().cpu().numpy())
            accum["pred_objective"].append(reconstructed["objective"].detach().cpu().numpy())
            accum["plan_row_ids"].extend(raw_batch["plan_row_id"])
            accum["workload_names"].extend(raw_batch["workload_name"])
            accum["context_ids"].extend(raw_batch["context_id"])
            accum["server_counts"].extend(raw_batch["server_count"])
            accum["utilizations"].extend(raw_batch["utilization"])

    arrays = {
        key: np.concatenate(value, axis=0)
        for key, value in accum.items()
        if key not in {
            "plan_row_ids", "workload_names", "context_ids", "server_counts", "utilizations"
        }
    }
    metrics = {
        f"{prefix}/loss/{name.replace('loss_', '')}": value / max(row_count, 1)
        for name, value in loss_sums.items()
    }
    metrics.update(_regression_metrics(arrays["true_logs"][:, 0], arrays["pred_logs"][:, 0], f"{prefix}/tracking/log_mean"))
    metrics.update(_regression_metrics(arrays["true_mean"], arrays["pred_mean"], f"{prefix}/tracking/mean_physical"))
    metrics.update(_regression_metrics(arrays["true_logs"][:, 1], arrays["pred_logs"][:, 1], f"{prefix}/tracking/log_p90"))
    metrics.update(_regression_metrics(arrays["true_p90"], arrays["pred_p90"], f"{prefix}/tracking/p90_physical"))
    metrics.update(_regression_metrics(arrays["true_qos"], arrays["pred_qos"], f"{prefix}/qos/per_job_probability"))
    metrics[f"{prefix}/qos/per_job_probability/brier"] = float(np.mean((arrays["pred_qos"] - arrays["true_qos"]) ** 2))
    metrics.update(_regression_metrics(arrays["true_max_qos"], arrays["pred_max_qos"], f"{prefix}/qos/max_probability"))
    metrics.update(_regression_metrics(arrays["true_m_rsr"], arrays["pred_m_rsr"], f"{prefix}/cost/M_RSR"))
    metrics.update(_regression_metrics(arrays["true_objective"], arrays["pred_objective"], f"{prefix}/cost/full_objective"))
    correlation = spearmanr(arrays["true_objective"], arrays["pred_objective"]).correlation
    metrics[f"{prefix}/cost/full_objective/spearman"] = float(correlation)

    true_tracking_pass = arrays["true_p90"] <= constants.tracking_threshold
    pred_tracking_pass = arrays["pred_p90"] <= constants.tracking_threshold
    true_qos_pass = arrays["true_max_qos"] <= constants.qos_threshold
    pred_qos_pass = arrays["pred_max_qos"] <= constants.qos_threshold
    true_combined = true_tracking_pass & true_qos_pass
    pred_combined = pred_tracking_pass & pred_qos_pass
    metrics.update(_classification_metrics(true_tracking_pass, pred_tracking_pass, f"{prefix}/feasibility/tracking"))
    metrics.update(_classification_metrics(true_qos_pass, pred_qos_pass, f"{prefix}/feasibility/qos"))
    metrics.update(_classification_metrics(true_combined, pred_combined, f"{prefix}/feasibility/combined"))

    p90_band = (arrays["true_p90"] >= 0.20) & (arrays["true_p90"] <= 0.40)
    if p90_band.any():
        metrics[f"{prefix}/boundary/p90_0p20_0p40/mae"] = float(np.mean(np.abs(arrays["pred_p90"][p90_band] - arrays["true_p90"][p90_band])))
        metrics[f"{prefix}/boundary/p90_0p20_0p40/constraint_flip_rate"] = float(np.mean(pred_tracking_pass[p90_band] != true_tracking_pass[p90_band]))
        metrics[f"{prefix}/boundary/p90_0p20_0p40/count"] = float(p90_band.sum())
    qos_band = (arrays["true_max_qos"] >= 0.05) & (arrays["true_max_qos"] <= 0.15)
    if qos_band.any():
        metrics[f"{prefix}/boundary/max_pj_0p05_0p15/mae"] = float(np.mean(np.abs(arrays["pred_max_qos"][qos_band] - arrays["true_max_qos"][qos_band])))
        metrics[f"{prefix}/boundary/max_pj_0p05_0p15/constraint_flip_rate"] = float(np.mean(pred_qos_pass[qos_band] != true_qos_pass[qos_band]))
        metrics[f"{prefix}/boundary/max_pj_0p05_0p15/count"] = float(qos_band.sum())

    workload_names = np.asarray(accum["workload_names"], dtype=object)
    if include_workload_metrics:
        for workload in sorted(set(workload_names)):
            subset = workload_names == workload
            metrics.update(
                _subset_summary_metrics(
                    true_p90=arrays["true_p90"],
                    pred_p90=arrays["pred_p90"],
                    true_max_qos=arrays["true_max_qos"],
                    pred_max_qos=arrays["pred_max_qos"],
                    mask=subset,
                    prefix=f"{prefix}/by_workload/{_slug(workload)}",
                    constants=constants,
                )
            )
        families = np.asarray([str(x).split("-")[0] for x in workload_names], dtype=object)
        for family in sorted(set(families)):
            subset = families == family
            metrics.update(
                _subset_summary_metrics(
                    true_p90=arrays["true_p90"],
                    pred_p90=arrays["pred_p90"],
                    true_max_qos=arrays["true_max_qos"],
                    pred_max_qos=arrays["pred_max_qos"],
                    mask=subset,
                    prefix=f"{prefix}/by_family/{_slug(family)}",
                    constants=constants,
                )
            )

    rows = None
    if return_rows:
        rows = pd.DataFrame(
            {
                "Plan_Row_ID": accum["plan_row_ids"],
                "Workload_Name": accum["workload_names"],
                "Context_ID": accum["context_ids"],
                "server_count": accum["server_counts"],
                "utilization": accum["utilizations"],
                "Actual_Mean_Tracking": arrays["true_mean"],
                "Predicted_Mean_Tracking": arrays["pred_mean"],
                "Actual_P90_Tracking": arrays["true_p90"],
                "Predicted_P90_Tracking": arrays["pred_p90"],
                "Actual_Max_Pj": arrays["true_max_qos"],
                "Predicted_Max_Pj": arrays["pred_max_qos"],
                "Actual_M_RSR": arrays["true_m_rsr"],
                "Predicted_M_RSR": arrays["pred_m_rsr"],
                "Actual_Full_Objective": arrays["true_objective"],
                "Predicted_Full_Objective": arrays["pred_objective"],
                "Actual_Tracking_Pass": true_tracking_pass,
                "Predicted_Tracking_Pass": pred_tracking_pass,
                "Actual_QoS_Pass": true_qos_pass,
                "Predicted_QoS_Pass": pred_qos_pass,
                "Actual_Both_Pass": true_combined,
                "Predicted_Both_Pass": pred_combined,
            }
        )
    return metrics, rows


def context_metrics_table(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for context_id, group in rows.groupby("Context_ID", sort=True):
        actual = group["Actual_Both_Pass"].astype(bool).to_numpy()
        predicted = group["Predicted_Both_Pass"].astype(bool).to_numpy()
        cls = _classification_metrics(actual, predicted, "x")
        output.append(
            {
                "Context_ID": context_id,
                "Rows": len(group),
                "Actual_Feasible": int(actual.sum()),
                "Predicted_Feasible": int(predicted.sum()),
                "Overlap": int((actual & predicted).sum()),
                "Feasible_Precision": cls["x/feasible_precision"],
                "Feasible_Recall": cls["x/actual_feasible_accuracy"],
                "Feasible_F1": cls["x/f1_feasible"],
                "P90_MAE": float(np.mean(np.abs(group["Predicted_P90_Tracking"] - group["Actual_P90_Tracking"]))),
                "Max_Pj_MAE": float(np.mean(np.abs(group["Predicted_Max_Pj"] - group["Actual_Max_Pj"]))),
                "Objective_MAE": float(np.mean(np.abs(group["Predicted_Full_Objective"] - group["Actual_Full_Objective"]))),
            }
        )
    return pd.DataFrame(output)


# ---------------------------------------------------------------------------
# Learning-rate schedule, checkpointing, training
# ---------------------------------------------------------------------------

def cosine_warmup_lr(
    epoch: int,
    *,
    max_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    decay_epochs = max(max_epochs - warmup_epochs - 1, 1)
    progress = min(max((epoch - warmup_epochs) / decay_epochs, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _checkpoint_payload(
    *,
    model,
    optimizer,
    epoch: int,
    data_metadata: BehaviorDataMetadata,
    model_config: dict,
    training_config: dict,
    metrics: dict,
    history: list[dict],
    best_scores: dict,
    best_epochs: dict,
    epochs_without_improvement: int,
    wandb_run=None,
) -> dict:
    return {
        "format_version": 3,
        "model_version": "flexdc_behavior_model",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "data_metadata": data_metadata.to_dict(),
        "training_config": training_config,
        "metrics_at_checkpoint": metrics,
        "history": history,
        "best_scores": best_scores,
        "best_epochs": best_epochs,
        "epochs_without_improvement": int(epochs_without_improvement),
        "wandb_run_id": getattr(wandb_run, "id", None),
        "wandb_run_name": getattr(wandb_run, "name", None),
    }


def save_training_checkpoint(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def _load_training_checkpoint(path: str | Path, *, model, optimizer, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if int(checkpoint.get("format_version", 0)) != 3:
        raise ValueError("Resume requires a behavior-model training checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_behavior_model(
    model,
    data: BehaviorDataBundle,
    *,
    epochs: int = 250,
    base_lr: float = 3e-4,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-5,
    device_name: str = "auto",
    mean_tracking_weight: float = 1.0,
    p90_tracking_weight: float = 1.0,
    qos_weight: float = 4.0,
    tracking_boundary_multiplier: float = 2.0,
    qos_boundary_multiplier: float = 2.0,
    checkpoint_dir: str | Path = "checkpoints",
    checkpoint_prefix: str = "flexdc_behavior_model",
    model_config: Mapping | None = None,
    training_config: Mapping | None = None,
    resume_from: str | Path | None = None,
    restore_role: str = "best_loss",
    wandb_run=None,
    metrics_every_n_epochs: int = 1,
    verbose: bool = True,
) -> tuple[object, TrainingResult]:
    if metrics_every_n_epochs != 1:
        raise ValueError("behavior-model checkpoint/early-stopping logic requires metrics_every_n_epochs=1")
    device = choose_device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "latest": checkpoint_dir / f"{checkpoint_prefix}_latest.pt",
        "best_loss": checkpoint_dir / f"{checkpoint_prefix}_best_loss.pt",
        "best_objective": checkpoint_dir / f"{checkpoint_prefix}_best_objective.pt",
        "best_feasibility": checkpoint_dir / f"{checkpoint_prefix}_best_feasibility.pt",
        "final": checkpoint_dir / f"{checkpoint_prefix}_final.pt",
    }

    history: list[dict] = []
    best_scores = {
        "validation_loss": float("inf"),
        "validation_objective_r2": -float("inf"),
        "validation_feasibility_f1": -float("inf"),
    }
    best_epochs = {"best_loss": None, "best_objective": None, "best_feasibility": None}
    epochs_without_improvement = 0
    start_epoch = 0
    if resume_from:
        checkpoint = _load_training_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        loaded_best_scores = dict(checkpoint.get("best_scores", {}))
        # Accept checkpoints produced before the heldout->validation rename.
        legacy_score_map = {
            "heldout_loss": "validation_loss",
            "heldout_objective_r2": "validation_objective_r2",
            "heldout_feasibility_f1": "validation_feasibility_f1",
        }
        for old_key, new_key in legacy_score_map.items():
            if old_key in loaded_best_scores and new_key not in loaded_best_scores:
                loaded_best_scores[new_key] = loaded_best_scores[old_key]
        best_scores.update(loaded_best_scores)
        best_epochs.update(checkpoint.get("best_epochs", {}))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        if verbose:
            print(f"Resuming from {resume_from} at epoch {start_epoch}.")

    start_time = time.time()
    stopped_early = False
    final_epoch = start_epoch - 1
    training_config_dict = dict(training_config or {})
    if model_config is not None:
        model_config_dict = dict(model_config)
    elif hasattr(getattr(model, "config", None), "to_dict"):
        model_config_dict = model.config.to_dict()
    else:
        model_config_dict = dict(getattr(model, "config", {}))

    for epoch in range(start_epoch, epochs):
        final_epoch = epoch
        epoch_start = time.time()
        current_lr = cosine_warmup_lr(
            epoch,
            max_epochs=epochs,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_epochs=warmup_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        if hasattr(data.train_loader.sampler, "set_epoch"):
            data.train_loader.sampler.set_epoch(epoch)

        model.train()
        gradient_norms = []
        for raw_batch in data.train_loader:
            batch = move_batch(raw_batch, device)
            output = model(batch["features"], batch["workload"], batch["mask"])
            loss, _ = behavior_loss(
                output,
                batch,
                data.metadata,
                mean_tracking_weight=mean_tracking_weight,
                p90_tracking_weight=p90_tracking_weight,
                qos_weight=qos_weight,
                tracking_boundary_multiplier=tracking_boundary_multiplier,
                qos_boundary_multiplier=qos_boundary_multiplier,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )
            gradient_norms.append(float(grad_norm.detach().cpu()))
            optimizer.step()

        train_metrics, _ = evaluate_behavior_loader(
            model,
            data.train_eval_loader,
            device=device,
            metadata=data.metadata,
            prefix="train",
            mean_tracking_weight=mean_tracking_weight,
            p90_tracking_weight=p90_tracking_weight,
            qos_weight=qos_weight,
            tracking_boundary_multiplier=tracking_boundary_multiplier,
            qos_boundary_multiplier=qos_boundary_multiplier,
            include_workload_metrics=False,
        )
        validation_metrics, _ = evaluate_behavior_loader(
            model,
            data.validation_loader,
            device=device,
            metadata=data.metadata,
            prefix="validation",
            mean_tracking_weight=mean_tracking_weight,
            p90_tracking_weight=p90_tracking_weight,
            qos_weight=qos_weight,
            tracking_boundary_multiplier=tracking_boundary_multiplier,
            qos_boundary_multiplier=qos_boundary_multiplier,
            include_workload_metrics=True,
        )
        row = {
            "epoch": int(epoch),
            "learning_rate": float(current_lr),
            "gradient_norm_mean": float(np.mean(gradient_norms)) if gradient_norms else float("nan"),
            "gradient_norm_max": float(np.max(gradient_norms)) if gradient_norms else float("nan"),
            "epoch_seconds": float(time.time() - epoch_start),
            **train_metrics,
            **validation_metrics,
        }
        history.append(row)

        validation_loss = float(validation_metrics["validation/loss/total"])
        objective_r2 = float(validation_metrics["validation/cost/full_objective/r2"])
        feasibility_f1 = float(validation_metrics["validation/feasibility/combined/f1_feasible"])
        improved_loss = validation_loss < best_scores["validation_loss"] - early_stopping_min_delta
        improved_objective = (
            np.isfinite(objective_r2)
            and objective_r2 > best_scores["validation_objective_r2"]
        )
        improved_feasibility = (
            np.isfinite(feasibility_f1)
            and feasibility_f1 > best_scores["validation_feasibility_f1"]
        )
        if improved_loss:
            best_scores["validation_loss"] = validation_loss
            best_epochs["best_loss"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if improved_objective:
            best_scores["validation_objective_r2"] = objective_r2
            best_epochs["best_objective"] = epoch
        if improved_feasibility:
            best_scores["validation_feasibility_f1"] = feasibility_f1
            best_epochs["best_feasibility"] = epoch

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            data_metadata=data.metadata,
            model_config=model_config_dict,
            training_config=training_config_dict,
            metrics=row,
            history=history,
            best_scores=best_scores,
            best_epochs=best_epochs,
            epochs_without_improvement=epochs_without_improvement,
            wandb_run=wandb_run,
        )
        save_training_checkpoint(paths["latest"], payload)
        if improved_loss:
            save_training_checkpoint(paths["best_loss"], payload)
        if improved_objective:
            save_training_checkpoint(paths["best_objective"], payload)
        if improved_feasibility:
            save_training_checkpoint(paths["best_feasibility"], payload)

        if wandb_run is not None:
            wandb_run.log(row, step=epoch)
        if verbose:
            print(
                f"Epoch {epoch:03d} | lr={current_lr:.3g} | "
                f"train={train_metrics['train/loss/total']:.6g} | "
                f"validation={validation_loss:.6g} | "
                f"feas P/R={validation_metrics['validation/feasibility/combined/feasible_precision']:.4f}/"
                f"{validation_metrics['validation/feasibility/combined/actual_feasible_accuracy']:.4f} | "
                f"obj R2={objective_r2:.4f} | patience={epochs_without_improvement}/{early_stopping_patience}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            if verbose:
                print(
                    f"Early stopping at epoch {epoch}; best validation loss was "
                    f"epoch {best_epochs['best_loss']} "
                    f"({best_scores['validation_loss']:.6g})."
                )
            break

    # Save the last reached state separately before restoring a best checkpoint.
    final_metrics = history[-1] if history else {}
    final_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=final_epoch,
        data_metadata=data.metadata,
        model_config=model_config_dict,
        training_config=training_config_dict,
        metrics=final_metrics,
        history=history,
        best_scores=best_scores,
        best_epochs=best_epochs,
        epochs_without_improvement=epochs_without_improvement,
        wandb_run=wandb_run,
    )
    save_training_checkpoint(paths["final"], final_payload)

    if restore_role not in paths:
        raise ValueError(f"Unknown restore_role={restore_role}; choose one of {sorted(paths)}")
    restore_path = paths[restore_role]
    if not restore_path.exists():
        restore_path = paths["best_loss"] if paths["best_loss"].exists() else paths["final"]
    restored = torch.load(restore_path, map_location=device, weights_only=False)
    model.load_state_dict(restored["model_state_dict"])

    elapsed = time.time() - start_time
    summary = {
        "training_seconds_this_call": float(elapsed),
        "start_epoch": int(start_epoch),
        "final_epoch": int(final_epoch),
        "stopped_early": bool(stopped_early),
        "best_scores": best_scores,
        "best_epochs": best_epochs,
        "restored_role": restore_role,
        "restored_checkpoint": str(restore_path),
    }
    if wandb_run is not None:
        wandb_run.summary["training_seconds_this_call"] = float(elapsed)
        wandb_run.summary["training_start_epoch"] = int(start_epoch)
        wandb_run.summary["training_final_epoch"] = int(final_epoch)
        wandb_run.summary["training_stopped_early"] = bool(stopped_early)
        wandb_run.summary["restored_checkpoint_role"] = str(restore_role)
        for name, value in best_scores.items():
            if np.isfinite(value):
                wandb_run.summary[f"best/{name}"] = float(value)
        for name, value in best_epochs.items():
            if value is not None:
                wandb_run.summary[f"best_epoch/{name}"] = int(value)
    if verbose:
        print(f"Training call completed in {elapsed:.1f} seconds on {device}.")
        print("Best epochs:", best_epochs)
        print("Restored:", restore_path)

    return model, TrainingResult(
        history=pd.DataFrame(history),
        summary=summary,
        checkpoint_paths={key: str(value) for key, value in paths.items()},
    )


def load_behavior_model_checkpoint(
    path: str | Path,
    *,
    model_class,
    config_class,
    device_name: str = "auto",
):
    device = choose_device(device_name)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = config_class(**checkpoint["model_config"])
    model = model_class(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def final_behavior_evaluation(
    model,
    data: BehaviorDataBundle,
    *,
    device_name: str = "auto",
    mean_tracking_weight: float = 1.0,
    p90_tracking_weight: float = 1.0,
    qos_weight: float = 4.0,
    tracking_boundary_multiplier: float = 2.0,
    qos_boundary_multiplier: float = 2.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate train and validation splits only.

    The test set is intentionally excluded so checkpoint selection and ordinary
    diagnostics cannot accidentally inspect it. Use
    :func:`final_behavior_test_evaluation` once after the checkpoint is locked.
    """
    device = choose_device(device_name)
    train_metrics, train_rows = evaluate_behavior_loader(
        model,
        data.train_eval_loader,
        device=device,
        metadata=data.metadata,
        prefix="train",
        mean_tracking_weight=mean_tracking_weight,
        p90_tracking_weight=p90_tracking_weight,
        qos_weight=qos_weight,
        tracking_boundary_multiplier=tracking_boundary_multiplier,
        qos_boundary_multiplier=qos_boundary_multiplier,
        return_rows=True,
        include_workload_metrics=True,
    )
    validation_metrics, validation_rows = evaluate_behavior_loader(
        model,
        data.validation_loader,
        device=device,
        metadata=data.metadata,
        prefix="validation",
        mean_tracking_weight=mean_tracking_weight,
        p90_tracking_weight=p90_tracking_weight,
        qos_weight=qos_weight,
        tracking_boundary_multiplier=tracking_boundary_multiplier,
        qos_boundary_multiplier=qos_boundary_multiplier,
        return_rows=True,
        include_workload_metrics=True,
    )
    assert train_rows is not None and validation_rows is not None
    return {**train_metrics, **validation_metrics}, train_rows, validation_rows


def final_behavior_test_evaluation(
    model,
    data: BehaviorDataBundle,
    *,
    device_name: str = "auto",
    mean_tracking_weight: float = 1.0,
    p90_tracking_weight: float = 1.0,
    qos_weight: float = 4.0,
    tracking_boundary_multiplier: float = 2.0,
    qos_boundary_multiplier: float = 2.0,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate the untouched test split once after model selection."""
    if data.test_loader is None or data.test_dataframe is None or len(data.test_dataframe) == 0:
        raise ValueError(
            "No test split is available. the frozen benchmark training requires a "
            "preassigned Data_Split column with train/validation/test rows."
        )
    device = choose_device(device_name)
    test_metrics, test_rows = evaluate_behavior_loader(
        model,
        data.test_loader,
        device=device,
        metadata=data.metadata,
        prefix="test",
        mean_tracking_weight=mean_tracking_weight,
        p90_tracking_weight=p90_tracking_weight,
        qos_weight=qos_weight,
        tracking_boundary_multiplier=tracking_boundary_multiplier,
        qos_boundary_multiplier=qos_boundary_multiplier,
        return_rows=True,
        include_workload_metrics=True,
    )
    assert test_rows is not None
    return test_metrics, test_rows


def sample_prediction_rows(rows: pd.DataFrame, max_rows: int = 1024, seed: int = 0) -> pd.DataFrame:
    if len(rows) <= max_rows:
        return rows.copy()
    return rows.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
