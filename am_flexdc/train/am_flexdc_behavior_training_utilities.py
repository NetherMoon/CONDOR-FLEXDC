"""Training utilities for the CONDOR Set-Transformer with FlexDC behavior labels.

Direct neural labels
--------------------
* log(mean normalized tracking error + 1e-6)
* log(p90 normalized tracking error + 1e-3)
* one per-job-type QoS violation probability P_j

Known FlexDC costs are reconstructed analytically for evaluation.  The model
never directly predicts M_RSR or the full objective.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


MEAN_TRACKING_LOG_FLOOR = 1e-6
P90_TRACKING_LOG_FLOOR = 1e-3
TRACKING_THRESHOLD = 0.3
QOS_THRESHOLD = 0.1
CTRACK_PSI = 1.0
CTRACK_MU = 10.0
QOS_BETA = 20.0
QOS_RHO = 2.0


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
    workload_norm_weights: list[float]
    log_mean_tracking_mean: float
    log_mean_tracking_std: float
    log_p90_tracking_mean: float
    log_p90_tracking_std: float
    feature_names: list[str]
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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BehaviorDataBundle:
    dataframe: pd.DataFrame
    train_dataset: "FlexDCBehaviorDataset"
    heldout_dataset: "FlexDCBehaviorDataset"
    train_loader: DataLoader
    train_eval_loader: DataLoader
    heldout_loader: DataLoader
    metadata: BehaviorDataMetadata
    audit: dict


MERGE_KEYS_PRIORITY = [
    ["Plan_Row_ID"],
    ["Source_Output_Dir", "Iteration"],
    ["Source_Output_Dir", "Iteration", "Weight_Sample_ID"],
]


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


def stable_softplus_np(x):
    return np.logaddexp(0.0, x)


def parse_list(value, column_name: str) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if pd.isna(value):
        raise ValueError(f"Missing list value in {column_name}")
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        cleaned = text.replace("np.float64(", "(")
        parsed = ast.literal_eval(cleaned)
    return [float(x) for x in parsed]


def parse_workload_mix(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(float, copy=True)
    else:
        try:
            arr = np.asarray(json.loads(str(value)), dtype=float)
        except Exception:
            arr = np.asarray(ast.literal_eval(str(value)), dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (6, 7):
        raise ValueError(f"workload_mix must have shape [J,6] or [J,7], got {arr.shape}")
    return arr


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def read_results_and_diagnostics(results_csv: str | Path, diagnostics_csv: str | Path | None) -> pd.DataFrame:
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
    if merged[extra_columns].isna().all(axis=1).any():
        raise ValueError("At least one results row failed to match diagnostics.")
    return merged


def canonical_vector(values: list[float], digits: int = 12) -> str:
    return json.dumps([round(float(x), digits) for x in values], separators=(",", ":"))


def _duplicate_identity_columns(df: pd.DataFrame) -> list[str]:
    required = [
        "Workload_Name",
        "server_count",
        "utilization",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "Simulation_Seed",
        "_weights_key",
    ]
    require_columns(df, required, "duplicate identity")
    return required


def deduplicate_behavior_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["_weights_key"] = df["weights"].apply(lambda x: canonical_vector(parse_list(x, "weights")))
    identity = _duplicate_identity_columns(df)
    duplicate_mask = df.duplicated(identity, keep=False)

    disagreement_groups = []
    if duplicate_mask.any():
        check_columns = [
            "Mtrack_Error_MeanAbs_Normalized",
            "Ctrack_Epsilon_90th",
            "Simulator_RSR_Total_Cost",
            "Diagnostic_FullPaperObjective_Cost",
        ]
        for _, group in df[duplicate_mask].groupby(identity, dropna=False):
            disagreement = False
            for column in check_columns:
                if column in group.columns and not np.allclose(
                    group[column].astype(float), group[column].astype(float).iloc[0], rtol=0, atol=1e-10
                ):
                    disagreement = True
            qos_keys = group["QoS_Delay_Probabilities"].apply(
                lambda x: canonical_vector(parse_list(x, "QoS_Delay_Probabilities"))
            )
            if qos_keys.nunique() != 1:
                disagreement = True
            if disagreement:
                disagreement_groups.append(group[identity + check_columns].to_dict(orient="records"))
    if disagreement_groups:
        raise ValueError(f"Duplicate input configurations have inconsistent labels: {disagreement_groups[:2]}")

    original_rows = len(df)
    df = df.drop_duplicates(identity, keep="first").reset_index(drop=True)
    removed = original_rows - len(df)
    audit = {
        "original_rows": int(original_rows),
        "deduplicated_rows": int(len(df)),
        "removed_duplicate_rows": int(removed),
        "duplicate_pair_rows_before_removal": int(duplicate_mask.sum()),
    }
    return df, audit


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
    weight_errors = []
    probability_errors = []
    for index, row in df.iterrows():
        mix = parse_workload_mix(row["workload_mix"])
        weights = np.asarray(parse_list(row["weights"], "weights"), dtype=float)
        probabilities = np.asarray(parse_list(row["QoS_Delay_Probabilities"], "QoS_Delay_Probabilities"), dtype=float)
        j = mix.shape[0]
        j_values.append(j)
        if int(row["workload_mix_size"]) != j or len(weights) != j:
            weight_errors.append(index)
        if len(probabilities) != j or not np.all((probabilities >= 0) & (probabilities <= 1)):
            probability_errors.append(index)
        if not np.isclose(weights.sum(), 1.0, atol=1e-8) or np.any(weights < 0):
            weight_errors.append(index)
    if weight_errors:
        raise ValueError(f"Invalid workload size/weights at rows: {sorted(set(weight_errors))[:20]}")
    if probability_errors:
        raise ValueError(f"Invalid QoS probability vectors at rows: {probability_errors[:20]}")

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
        "server_counts": sorted(int(x) for x in df["server_count"].unique()),
        "utilizations": sorted(float(x) for x in df["utilization"].unique()),
        "mean_tracking_min": float(mean_tracking.min()),
        "mean_tracking_max": float(mean_tracking.max()),
        "p90_tracking_min": float(p90_tracking.min()),
        "p90_tracking_max": float(p90_tracking.max()),
    }


def compute_workload_norm_weights(df: pd.DataFrame) -> np.ndarray:
    total = np.zeros(6, dtype=float)
    count = 0
    for value in df["workload_mix"]:
        arr = parse_workload_mix(value)[:, :6]
        total += np.abs(arr).sum(axis=0)
        count += arr.shape[0]
    means = total / max(count, 1)
    means[means == 0] = 1.0
    return np.concatenate([means, np.asarray([1.0])])


def _workload_tensor(row: pd.Series, norm_weights: np.ndarray, use_norm_wlmix: bool) -> torch.Tensor:
    base = parse_workload_mix(row["workload_mix"])
    weights = np.asarray(parse_list(row["weights"], "weights"), dtype=float)
    if base.shape[1] == 6:
        base = np.concatenate([base, weights.reshape(-1, 1)], axis=1)
    else:
        base = base.copy()
        base[:, 6] = weights
    if use_norm_wlmix:
        base = base / norm_weights
    return torch.tensor(base, dtype=torch.float32)


def _group_id_for_row(row: pd.Series) -> str:
    if "Base_Plan_Row_ID" in row.index and not pd.isna(row["Base_Plan_Row_ID"]):
        return str(row["Base_Plan_Row_ID"])
    pieces = [
        str(row["Workload_Name"]),
        str(int(row["server_count"])),
        f"{float(row['utilization']):.12g}",
        f"{float(row['Pbar_kw_per_server']):.12g}",
        f"{float(row['R_kw_per_server']):.12g}",
        canonical_vector(parse_list(row["weights"], "weights")),
    ]
    return hashlib.sha256("|".join(pieces).encode()).hexdigest()[:24]


class FlexDCBehaviorDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        workload_norm_weights: np.ndarray,
        use_norm_pr: bool = True,
        use_norm_wlmix: bool = True,
        constants: FlexDCBehaviorConstants | None = None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.workload_norm_weights = np.asarray(workload_norm_weights, dtype=float)
        self.use_norm_pr = bool(use_norm_pr)
        self.use_norm_wlmix = bool(use_norm_wlmix)
        self.constants = constants or FlexDCBehaviorConstants()

        if use_norm_pr:
            require_columns(self.df, ["Pbar_ratio", "R_ratio"], "normalized P/R features")
            p = self.df["Pbar_ratio"].astype(float).to_numpy()
            r = self.df["R_ratio"].astype(float).to_numpy()
            p_name, r_name = "Pbar_ratio", "R_ratio"
        else:
            p = self.df["Pbar_kw_per_server"].astype(float).to_numpy()
            r = self.df["R_kw_per_server"].astype(float).to_numpy()
            p_name, r_name = "Pbar_kw_per_server", "R_kw_per_server"

        features = np.zeros((len(self.df), 5), dtype=np.float32)
        features[:, 0] = p
        features[:, 1] = r
        features[:, 2] = self.df["server_count"].astype(float).to_numpy()
        features[:, 3] = self.df["utilization"].astype(float).to_numpy()
        features[:, 4] = self.df["workload_mix_size"].astype(float).to_numpy()
        self.features = torch.tensor(features, dtype=torch.float32)
        self.feature_names = [p_name, r_name, "server_count", "utilization", "workload_mix_size"]

        self.workloads = [
            _workload_tensor(row, self.workload_norm_weights, self.use_norm_wlmix)
            for _, row in self.df.iterrows()
        ]
        self.qos_targets = [
            torch.tensor(parse_list(value, "QoS_Delay_Probabilities"), dtype=torch.float32)
            for value in self.df["QoS_Delay_Probabilities"]
        ]
        mean_tracking = self.df["Mtrack_Error_MeanAbs_Normalized"].astype(float).to_numpy()
        p90_tracking = self.df["Ctrack_Epsilon_90th"].astype(float).to_numpy()
        self.log_tracking = torch.tensor(
            np.stack(
                [
                    np.log(mean_tracking + self.constants.mean_tracking_log_floor),
                    np.log(p90_tracking + self.constants.p90_tracking_log_floor),
                ],
                axis=1,
            ),
            dtype=torch.float32,
        )
        self.raw_mean_tracking = torch.tensor(mean_tracking, dtype=torch.float32)
        self.raw_p90_tracking = torch.tensor(p90_tracking, dtype=torch.float32)

        self.simulator_power_cost = torch.tensor(
            self.df["Simulator_Power_Cost"].astype(float).to_numpy(), dtype=torch.float32
        )
        self.actual_m_rsr = torch.tensor(
            self.df["Simulator_RSR_Total_Cost"].astype(float).to_numpy(), dtype=torch.float32
        )
        self.actual_objective = torch.tensor(
            self.df["Diagnostic_FullPaperObjective_Cost"].astype(float).to_numpy(), dtype=torch.float32
        )
        self.r_actual_watts = torch.tensor(
            self.df["R_actual_watts"].astype(float).to_numpy(), dtype=torch.float32
        )
        self.tracking_price_coefficient = torch.tensor(
            self.df["Mtrack_Price_Coefficient_piE"].astype(float).to_numpy(), dtype=torch.float32
        )
        self.hour_seconds = torch.tensor(
            self.df["Mtrack_Hour_Seconds"].astype(float).to_numpy(), dtype=torch.float32
        )
        self.plan_row_ids = (
            self.df["Plan_Row_ID"].astype(str).tolist()
            if "Plan_Row_ID" in self.df.columns
            else [str(index) for index in range(len(self.df))]
        )

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
    }


def prepare_behavior_data(
    *,
    results_csv: str | Path,
    diagnostics_csv: str | Path | None,
    batch_size: int = 512,
    heldout_fraction: float = 0.30,
    split_seed: int = 0,
    use_norm_pr: bool = True,
    use_norm_wlmix: bool = True,
    num_workers: int = 0,
    deduplicate: bool = True,
    constants: FlexDCBehaviorConstants | None = None,
) -> BehaviorDataBundle:
    constants = constants or FlexDCBehaviorConstants()
    merged = read_results_and_diagnostics(results_csv, diagnostics_csv)
    original_rows = len(merged)
    if deduplicate:
        merged, duplicate_audit = deduplicate_behavior_rows(merged)
    else:
        duplicate_audit = {
            "original_rows": original_rows,
            "deduplicated_rows": original_rows,
            "removed_duplicate_rows": 0,
            "duplicate_pair_rows_before_removal": 0,
        }
    schema_audit = validate_behavior_dataframe(merged)
    merged["_group_id"] = merged.apply(_group_id_for_row, axis=1)

    groups = np.asarray(sorted(merged["_group_id"].unique()))
    train_groups, heldout_groups = train_test_split(
        groups,
        test_size=heldout_fraction,
        random_state=split_seed,
        shuffle=True,
    )
    train_set = set(train_groups)
    heldout_set = set(heldout_groups)
    train_df = merged[merged["_group_id"].isin(train_set)].copy().reset_index(drop=True)
    heldout_df = merged[merged["_group_id"].isin(heldout_set)].copy().reset_index(drop=True)
    if set(train_df["_group_id"]) & set(heldout_df["_group_id"]):
        raise AssertionError("Grouped split leakage detected.")

    workload_norm = compute_workload_norm_weights(train_df) if use_norm_wlmix else np.ones(7)
    train_mean = train_df["Mtrack_Error_MeanAbs_Normalized"].astype(float).to_numpy()
    train_p90 = train_df["Ctrack_Epsilon_90th"].astype(float).to_numpy()
    log_mean = np.log(train_mean + constants.mean_tracking_log_floor)
    log_p90 = np.log(train_p90 + constants.p90_tracking_log_floor)
    log_mean_std = float(np.std(log_mean)) or 1.0
    log_p90_std = float(np.std(log_p90)) or 1.0

    train_dataset = FlexDCBehaviorDataset(
        train_df,
        workload_norm_weights=workload_norm,
        use_norm_pr=use_norm_pr,
        use_norm_wlmix=use_norm_wlmix,
        constants=constants,
    )
    heldout_dataset = FlexDCBehaviorDataset(
        heldout_df,
        workload_norm_weights=workload_norm,
        use_norm_pr=use_norm_pr,
        use_norm_wlmix=use_norm_wlmix,
        constants=constants,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_behavior_batch,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    heldout_loader = DataLoader(heldout_dataset, shuffle=False, **loader_kwargs)

    metadata = BehaviorDataMetadata(
        workload_norm_weights=workload_norm.tolist(),
        log_mean_tracking_mean=float(np.mean(log_mean)),
        log_mean_tracking_std=log_mean_std,
        log_p90_tracking_mean=float(np.mean(log_p90)),
        log_p90_tracking_std=log_p90_std,
        feature_names=train_dataset.feature_names,
        direct_label_names=[
            "log_mean_tracking",
            "log_p90_tracking",
            "P_j_per_real_job_type",
        ],
        constants=constants.to_dict(),
        split_strategy="grouped_Base_Plan_Row_ID_70_30",
        split_seed=int(split_seed),
        train_group_count=int(len(train_groups)),
        heldout_group_count=int(len(heldout_groups)),
        train_row_count=int(len(train_df)),
        heldout_row_count=int(len(heldout_df)),
        original_row_count=int(original_rows),
        deduplicated_row_count=int(len(merged)),
        removed_duplicate_rows=int(duplicate_audit["removed_duplicate_rows"]),
    )
    audit = {
        **duplicate_audit,
        **schema_audit,
        "split_strategy": metadata.split_strategy,
        "train_rows": len(train_df),
        "heldout_rows": len(heldout_df),
        "train_groups": len(train_groups),
        "heldout_groups": len(heldout_groups),
        "group_overlap": 0,
    }
    return BehaviorDataBundle(
        dataframe=merged,
        train_dataset=train_dataset,
        heldout_dataset=heldout_dataset,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        heldout_loader=heldout_loader,
        metadata=metadata,
        audit=audit,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def decode_tracking_logs(logs: torch.Tensor, constants: FlexDCBehaviorConstants) -> tuple[torch.Tensor, torch.Tensor]:
    mean_tracking = torch.clamp(torch.exp(logs[:, 0]) - constants.mean_tracking_log_floor, min=0.0)
    p90_tracking = torch.clamp(torch.exp(logs[:, 1]) - constants.p90_tracking_log_floor, min=0.0)
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
    ctrack = constants.ctrack_psi * F.softplus(constants.ctrack_mu * (p90_tracking - constants.tracking_threshold))
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
    qos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_logs = model_output["tracking_logs"]
    true_logs = batch["tracking_logs"]
    mean_std = max(metadata.log_mean_tracking_std, 1e-12)
    p90_std = max(metadata.log_p90_tracking_std, 1e-12)
    mean_loss = F.smooth_l1_loss(
        predicted_logs[:, 0] / mean_std,
        true_logs[:, 0] / mean_std,
    )
    p90_loss = F.smooth_l1_loss(
        predicted_logs[:, 1] / p90_std,
        true_logs[:, 1] / p90_std,
    )
    mask = batch["mask"].to(predicted_logs.dtype)
    squared_qos_error = (model_output["qos_probabilities"] - batch["qos_probabilities"]) ** 2
    qos_loss = (squared_qos_error * mask).sum() / mask.sum().clamp_min(1.0)
    total = mean_tracking_weight * mean_loss + p90_tracking_weight * p90_loss + qos_weight * qos_loss
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
    return {
        f"{prefix}/accuracy_overall": (tp + tn) / total if total else float("nan"),
        # Fatih-requested class-specific accuracies:
        f"{prefix}/actual_feasible_accuracy": tp / actual_feasible if actual_feasible else float("nan"),
        f"{prefix}/actual_infeasible_accuracy": tn / actual_infeasible if actual_infeasible else float("nan"),
        f"{prefix}/feasible_precision": tp / predicted_feasible if predicted_feasible else float("nan"),
        f"{prefix}/false_feasible_rate": fp / actual_infeasible if actual_infeasible else float("nan"),
        f"{prefix}/false_infeasible_rate": fn / actual_feasible if actual_feasible else float("nan"),
        f"{prefix}/true_feasible_count": float(tp),
        f"{prefix}/true_infeasible_count": float(tn),
        f"{prefix}/false_feasible_count": float(fp),
        f"{prefix}/false_infeasible_count": float(fn),
        f"{prefix}/actual_feasible_count": float(actual_feasible),
        f"{prefix}/actual_infeasible_count": float(actual_infeasible),
    }


def evaluate_behavior_loader(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    metadata: BehaviorDataMetadata,
    prefix: str,
    mean_tracking_weight: float = 1.0,
    p90_tracking_weight: float = 1.0,
    qos_weight: float = 1.0,
    return_rows: bool = False,
) -> tuple[dict, pd.DataFrame | None]:
    model.eval()
    constants = FlexDCBehaviorConstants(**metadata.constants)
    accum = {
        "true_logs": [], "pred_logs": [],
        "true_mean": [], "pred_mean": [],
        "true_p90": [], "pred_p90": [],
        "true_qos": [], "pred_qos": [],
        "true_max_qos": [], "pred_max_qos": [],
        "true_m_rsr": [], "pred_m_rsr": [],
        "true_objective": [], "pred_objective": [],
        "plan_row_ids": [],
    }
    loss_sums = {"loss_total": 0.0, "loss_log_mean_tracking_huber": 0.0, "loss_log_p90_tracking_huber": 0.0, "loss_qos_brier": 0.0}
    row_count = 0

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(batch["features"], batch["workload"], batch["mask"])
            _, losses = behavior_loss(
                output, batch, metadata,
                mean_tracking_weight=mean_tracking_weight,
                p90_tracking_weight=p90_tracking_weight,
                qos_weight=qos_weight,
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
            mask = batch["mask"].detach().cpu().numpy().astype(bool)
            true_qos = batch["qos_probabilities"].detach().cpu().numpy()
            pred_qos = output["qos_probabilities"].detach().cpu().numpy()
            accum["true_logs"].append(batch["tracking_logs"].detach().cpu().numpy())
            accum["pred_logs"].append(output["tracking_logs"].detach().cpu().numpy())
            accum["true_mean"].append(batch["raw_mean_tracking"].detach().cpu().numpy())
            accum["pred_mean"].append(reconstructed["mean_tracking"].detach().cpu().numpy())
            accum["true_p90"].append(batch["raw_p90_tracking"].detach().cpu().numpy())
            accum["pred_p90"].append(reconstructed["p90_tracking"].detach().cpu().numpy())
            accum["true_qos"].append(true_qos[mask])
            accum["pred_qos"].append(pred_qos[mask])
            true_masked = np.where(mask, true_qos, -np.inf)
            pred_masked = np.where(mask, pred_qos, -np.inf)
            accum["true_max_qos"].append(np.max(true_masked, axis=1))
            accum["pred_max_qos"].append(np.max(pred_masked, axis=1))
            accum["true_m_rsr"].append(batch["actual_m_rsr"].detach().cpu().numpy())
            accum["pred_m_rsr"].append(reconstructed["m_rsr"].detach().cpu().numpy())
            accum["true_objective"].append(batch["actual_objective"].detach().cpu().numpy())
            accum["pred_objective"].append(reconstructed["objective"].detach().cpu().numpy())
            accum["plan_row_ids"].extend(raw_batch["plan_row_id"])

    arrays = {
        key: np.concatenate(value, axis=0)
        for key, value in accum.items()
        if key != "plan_row_ids"
    }
    metrics = {f"{prefix}/loss/{name.replace('loss_', '')}": value / max(row_count, 1) for name, value in loss_sums.items()}
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

    rows = None
    if return_rows:
        rows = pd.DataFrame({
            "Plan_Row_ID": accum["plan_row_ids"],
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
        })
    return metrics, rows


def train_behavior_model(
    model,
    data: BehaviorDataBundle,
    *,
    epochs: int = 150,
    lr: float = 1e-4,
    device_name: str = "auto",
    mean_tracking_weight: float = 1.0,
    p90_tracking_weight: float = 1.0,
    qos_weight: float = 1.0,
    wandb_run=None,
    metrics_every_n_epochs: int = 1,
    verbose: bool = True,
) -> tuple[object, pd.DataFrame]:
    device = choose_device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running = {"total": 0.0, "mean": 0.0, "p90": 0.0, "qos": 0.0, "rows": 0}
        for raw_batch in data.train_loader:
            batch = move_batch(raw_batch, device)
            output = model(batch["features"], batch["workload"], batch["mask"])
            loss, components = behavior_loss(
                output,
                batch,
                data.metadata,
                mean_tracking_weight=mean_tracking_weight,
                p90_tracking_weight=p90_tracking_weight,
                qos_weight=qos_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            size = batch["features"].size(0)
            running["rows"] += size
            running["total"] += float(components["loss_total"].detach().cpu()) * size
            running["mean"] += float(components["loss_log_mean_tracking_huber"].detach().cpu()) * size
            running["p90"] += float(components["loss_log_p90_tracking_huber"].detach().cpu()) * size
            running["qos"] += float(components["loss_qos_brier"].detach().cpu()) * size

        should_evaluate = (epoch % metrics_every_n_epochs == 0) or epoch == epochs - 1
        if should_evaluate:
            train_metrics, _ = evaluate_behavior_loader(
                model,
                data.train_eval_loader,
                device=device,
                metadata=data.metadata,
                prefix="train",
                mean_tracking_weight=mean_tracking_weight,
                p90_tracking_weight=p90_tracking_weight,
                qos_weight=qos_weight,
            )
            heldout_metrics, _ = evaluate_behavior_loader(
                model,
                data.heldout_loader,
                device=device,
                metadata=data.metadata,
                prefix="heldout",
                mean_tracking_weight=mean_tracking_weight,
                p90_tracking_weight=p90_tracking_weight,
                qos_weight=qos_weight,
            )
            row = {"epoch": epoch, **train_metrics, **heldout_metrics}
            history.append(row)
            if wandb_run is not None:
                wandb_run.log(row, step=epoch)
            if verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train loss={train_metrics['train/loss/total']:.6g} | "
                    f"heldout loss={heldout_metrics['heldout/loss/total']:.6g} | "
                    f"heldout feasible acc={heldout_metrics['heldout/feasibility/combined/accuracy_overall']:.4f} | "
                    f"feasible-class acc={heldout_metrics['heldout/feasibility/combined/actual_feasible_accuracy']:.4f} | "
                    f"infeasible-class acc={heldout_metrics['heldout/feasibility/combined/actual_infeasible_accuracy']:.4f}"
                )

    elapsed = time.time() - start_time
    if wandb_run is not None:
        wandb_run.summary["training_seconds"] = elapsed
    if verbose:
        print(f"Training completed in {elapsed:.1f} seconds on {device}.")
    return model, pd.DataFrame(history)


def final_behavior_evaluation(
    model,
    data: BehaviorDataBundle,
    *,
    device_name: str = "auto",
    mean_tracking_weight: float = 1.0,
    p90_tracking_weight: float = 1.0,
    qos_weight: float = 1.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
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
        return_rows=True,
    )
    heldout_metrics, heldout_rows = evaluate_behavior_loader(
        model,
        data.heldout_loader,
        device=device,
        metadata=data.metadata,
        prefix="heldout",
        mean_tracking_weight=mean_tracking_weight,
        p90_tracking_weight=p90_tracking_weight,
        qos_weight=qos_weight,
        return_rows=True,
    )
    return {**train_metrics, **heldout_metrics}, train_rows, heldout_rows


def save_behavior_checkpoint(
    path: str | Path,
    *,
    model,
    data_metadata: BehaviorDataMetadata,
    model_config: dict,
    training_config: dict,
    metrics: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "data_metadata": data_metadata.to_dict(),
            "training_config": training_config,
            "final_metrics": metrics,
        },
        path,
    )
    return path


def sample_prediction_rows(rows: pd.DataFrame, max_rows: int = 1024, seed: int = 0) -> pd.DataFrame:
    if len(rows) <= max_rows:
        return rows.copy()
    return rows.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
