"""Unified CONDOR/FlexDC training utilities.

This file intentionally keeps the same high-level structure as CONDOR's
original training_utilities.py:

    DCDataset -> train_model -> evaluate_model

The changes are limited to the dataset loader and optional W&B/device plumbing.
The DataCenterModel architecture is unchanged and remains in data_center_model.py.

Supported target variants
-------------------------
1. target_family='condor', target_mode='normal'
   Targets are CONDOR/AQA cost components reconstructed from FlexDC simulator
   outputs, then scaled like the released CONDOR loader:
       cost_power = 0.0003 * (P_actual_watts - R_actual_watts)
       cost_error = Mtrack_Error_MeanAbs_Watts / 1000
       cost_qos   = 0.8 * sum(SoftPlus(60 * (QoS_probability - 0.1)))
       y_power    = 120 * cost_power / server_count
       y_error    = 200 * cost_error / server_count
       y_qos      = cost_qos / workload_mix_size

2. target_family='condor', target_mode='raw'
   Keeps CONDOR power/error components but replaces the SoftPlus QoS label with
   the measured QoS delay probability aggregate. With use_norm_cost=True, the
   QoS target is the per-job-type average probability.

3. target_family='flexdc', target_mode='normal'
   Targets are the FlexDC paper-style components logged by the data wizard:
       M_RSR, weighted Ctrack, weighted CQoS.

4. target_family='flexdc', target_mode='raw'
   Keeps M_RSR and replaces FlexDC SoftPlus terms with raw measured quantities:
       Ctrack_Epsilon_90th, QoS delay probability aggregate.

Notes
-----
- The loader accepts results CSV + optional diagnostics CSV, or a single merged
  CSV that already contains diagnostics columns.
- By default P/R inputs use Pbar_ratio and R_ratio, matching the original CONDOR
  normalized P/R input convention and keeping inputs the same across all target
  variants.
- Workload feature normalization is recomputed from the FlexDC dataset, instead
  of reusing the released CONDOR constants from its original workload set.
"""

from __future__ import annotations

import ast
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


CONDOR_POWER_COST_COEFFICIENT = 3e-4
CONDOR_QOS_BETA = 0.8
CONDOR_QOS_RHO = 60.0
CONDOR_QOS_THRESHOLD = 0.1

TARGET_FAMILIES = {"condor", "flexdc"}
TARGET_MODES = {"normal", "raw"}
RAW_QOS_AGGREGATIONS = {"mean", "sum"}

MERGE_KEYS = [
    "Source_Output_Dir",
    "Iteration",
    "Weight_Sample_ID",
    "Workload_Name",
    "Workload_Config",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "server_count",
    "utilization",
]


@dataclass(frozen=True)
class TargetSpec:
    target_family: str
    target_mode: str
    use_norm_cost: bool
    raw_qos_aggregation: str
    target_names: list[str]


def choose_device(device_name: str = "auto") -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device option: {device_name}")


def softplus_np(x):
    return np.logaddexp(0, x)


def parse_json_list(value, column_name: str) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if pd.isna(value):
        raise ValueError(f"Missing JSON/list value in column {column_name}.")
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    return [float(x) for x in parsed]


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def read_results_and_diagnostics(results_csv: str | Path, diagnostics_csv: str | Path | None = None) -> pd.DataFrame:
    results = pd.read_csv(results_csv)
    if diagnostics_csv is None or str(diagnostics_csv).strip() == "":
        return results

    diagnostics = pd.read_csv(diagnostics_csv)
    keys = [key for key in MERGE_KEYS if key in results.columns and key in diagnostics.columns]

    if keys:
        extra_cols = [col for col in diagnostics.columns if col not in results.columns]
        merged = results.merge(
            diagnostics[keys + extra_cols],
            on=keys,
            how="left",
            validate="one_to_one",
        )
        if len(merged) != len(results):
            raise ValueError(
                f"Diagnostics merge changed row count: results={len(results)}, merged={len(merged)}"
            )
        return merged

    if len(results) != len(diagnostics):
        raise ValueError(
            "Could not find merge keys shared by results and diagnostics, and row counts differ: "
            f"results={len(results)}, diagnostics={len(diagnostics)}"
        )

    extra_cols = [col for col in diagnostics.columns if col not in results.columns]
    return pd.concat([results.reset_index(drop=True), diagnostics[extra_cols].reset_index(drop=True)], axis=1)


def get_first_column(df: pd.DataFrame, candidates: list[str], context: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find any of {candidates} for {context}.")


def get_p_actual_watts(df: pd.DataFrame) -> np.ndarray:
    col = get_first_column(df, ["P_actual_watts", "P_actual"], "actual P watts")
    return df[col].astype(float).to_numpy()


def get_r_actual_watts(df: pd.DataFrame) -> np.ndarray:
    col = get_first_column(df, ["R_actual_watts", "R_actual"], "actual R watts")
    return df[col].astype(float).to_numpy()


def qos_probability_matrix(df: pd.DataFrame) -> np.ndarray:
    require_columns(df, ["QoS_Delay_Probabilities"], "QoS probability parsing")
    probs = [parse_json_list(x, "QoS_Delay_Probabilities") for x in df["QoS_Delay_Probabilities"]]
    lengths = {len(row) for row in probs}
    if len(lengths) != 1:
        raise ValueError(f"QoS_Delay_Probabilities have inconsistent lengths: {sorted(lengths)}")
    return np.asarray(probs, dtype=float)


def qos_probability_aggregate(df: pd.DataFrame, aggregation: str) -> np.ndarray:
    if aggregation not in RAW_QOS_AGGREGATIONS:
        raise ValueError(f"raw_qos_aggregation must be one of {sorted(RAW_QOS_AGGREGATIONS)}")
    probs = qos_probability_matrix(df)
    if aggregation == "sum":
        return probs.sum(axis=1)
    return probs.mean(axis=1)


def condor_targets(df: pd.DataFrame, target_mode: str, use_norm_cost: bool, raw_qos_aggregation: str):
    require_columns(
        df,
        ["server_count", "workload_mix_size", "Mtrack_Error_MeanAbs_Watts", "QoS_Delay_Probabilities"],
        "CONDOR target construction",
    )

    server_counts = df["server_count"].astype(float).to_numpy()
    workload_mix_size = df["workload_mix_size"].astype(float).to_numpy()
    p_actual = get_p_actual_watts(df)
    r_actual = get_r_actual_watts(df)

    raw_power = CONDOR_POWER_COST_COEFFICIENT * (p_actual - r_actual)
    raw_error = df["Mtrack_Error_MeanAbs_Watts"].astype(float).to_numpy() / 1000.0

    if target_mode == "normal":
        probabilities = qos_probability_matrix(df)
        raw_qos = CONDOR_QOS_BETA * softplus_np(
            CONDOR_QOS_RHO * (probabilities - CONDOR_QOS_THRESHOLD)
        ).sum(axis=1)
        names = ["condor_cost_power", "condor_cost_error", "condor_cost_qos"]
    elif target_mode == "raw":
        raw_qos = qos_probability_aggregate(df, raw_qos_aggregation)
        names = ["condor_cost_power", "condor_cost_error", f"raw_qos_probability_{raw_qos_aggregation}"]
    else:
        raise ValueError(f"Unknown target_mode for CONDOR: {target_mode}")

    if use_norm_cost:
        # Released CONDOR code scaling. The QoS raw variant uses mean probability
        # when raw_qos_aggregation='mean', so dividing again is intentionally not done.
        y_power = raw_power * 120.0 / server_counts
        y_error = raw_error * 200.0 / server_counts
        if target_mode == "normal":
            y_qos = raw_qos / workload_mix_size
        else:
            y_qos = raw_qos if raw_qos_aggregation == "mean" else raw_qos / workload_mix_size
    else:
        y_power, y_error, y_qos = raw_power, raw_error, raw_qos

    targets = np.stack([y_power, y_error, y_qos], axis=1)
    return targets, names


def flexdc_targets(df: pd.DataFrame, target_mode: str, raw_qos_aggregation: str):
    if target_mode == "normal":
        require_columns(
            df,
            [
                "Simulator_RSR_Total_Cost",
                "Ctrack_Weighted_Cost",
                "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
            ],
            "FlexDC normal target construction",
        )
        targets = np.stack(
            [
                df["Simulator_RSR_Total_Cost"].astype(float).to_numpy(),
                df["Ctrack_Weighted_Cost"].astype(float).to_numpy(),
                df["Diagnostic_FlexDC_SoftPlus_QoS_Cost"].astype(float).to_numpy(),
            ],
            axis=1,
        )
        names = ["flexdc_M_RSR", "flexdc_Ctrack_weighted", "flexdc_CQoS_weighted"]

        if "Diagnostic_FullPaperObjective_Cost" in df.columns:
            recorded_total = df["Diagnostic_FullPaperObjective_Cost"].astype(float).to_numpy()
            if not np.allclose(recorded_total, targets.sum(axis=1), rtol=1e-6, atol=1e-6):
                raise ValueError(
                    "FlexDC normal targets do not reconstruct Diagnostic_FullPaperObjective_Cost."
                )

    elif target_mode == "raw":
        require_columns(
            df,
            ["Simulator_RSR_Total_Cost", "Ctrack_Epsilon_90th", "QoS_Delay_Probabilities"],
            "FlexDC raw target construction",
        )
        targets = np.stack(
            [
                df["Simulator_RSR_Total_Cost"].astype(float).to_numpy(),
                df["Ctrack_Epsilon_90th"].astype(float).to_numpy(),
                qos_probability_aggregate(df, raw_qos_aggregation),
            ],
            axis=1,
        )
        names = ["flexdc_M_RSR", "raw_Ctrack_Epsilon_90th", f"raw_qos_probability_{raw_qos_aggregation}"]
    else:
        raise ValueError(f"Unknown target_mode for FlexDC: {target_mode}")

    return targets, names


def build_targets(df: pd.DataFrame, target_family: str, target_mode: str, use_norm_cost: bool, raw_qos_aggregation: str):
    target_family = target_family.lower().strip()
    target_mode = target_mode.lower().strip()
    if target_family not in TARGET_FAMILIES:
        raise ValueError(f"target_family must be one of {sorted(TARGET_FAMILIES)}")
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {sorted(TARGET_MODES)}")

    if target_family == "condor":
        return condor_targets(df, target_mode, use_norm_cost, raw_qos_aggregation)
    return flexdc_targets(df, target_mode, raw_qos_aggregation)


def parse_workload_mix(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
    else:
        text = str(value).strip()
        try:
            arr = np.asarray(json.loads(text), dtype=float)
        except json.JSONDecodeError:
            arr = np.asarray(ast.literal_eval(text), dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"workload_mix must be a 2-D array; got shape {arr.shape}")
    if arr.shape[1] not in (6, 7):
        raise ValueError(
            "workload_mix rows must have six or seven values: "
            "[pmin, pmax, Tmin, Tmax, qos, nodes] plus optional weight. "
            f"Got shape {arr.shape}."
        )
    return arr


def weight_vector_from_row(row: pd.Series, workload_size: int) -> np.ndarray:
    weight_cols = [col for col in row.index if str(col).startswith("Weight_") and str(col) != "Weight_Sample_ID"]
    parsed_cols = []
    for col in weight_cols:
        suffix = str(col).split("_")[-1]
        if suffix.isdigit() and not pd.isna(row[col]):
            parsed_cols.append((int(suffix), col))
    if parsed_cols:
        parsed_cols = sorted(parsed_cols)
        weights = np.asarray([float(row[col]) for _, col in parsed_cols], dtype=float)
    elif "weights" in row.index and not pd.isna(row["weights"]):
        weights = np.asarray(parse_json_list(row["weights"], "weights"), dtype=float)
    else:
        weights = None

    if weights is None:
        return weights
    if len(weights) != workload_size:
        raise ValueError(f"Weight vector length {len(weights)} does not match workload size {workload_size}.")
    return weights


def extract_weight_matrix(df: pd.DataFrame, workload_size: int) -> np.ndarray | None:
    weight_cols = []
    for col in df.columns:
        text = str(col)
        if text.startswith("Weight_") and text != "Weight_Sample_ID":
            suffix = text.split("_")[-1]
            if suffix.isdigit():
                weight_cols.append((int(suffix), col))
    if weight_cols:
        weight_cols = [col for _, col in sorted(weight_cols)]
        weights = df[weight_cols].astype(float).to_numpy()
        if weights.shape[1] != workload_size:
            raise ValueError(
                f"Found {weights.shape[1]} Weight_i columns, but workload size is {workload_size}."
            )
        return weights
    if "weights" in df.columns:
        parsed = [parse_json_list(x, "weights") for x in df["weights"]]
        weights = np.asarray(parsed, dtype=float)
        if weights.shape[1] != workload_size:
            raise ValueError(f"weights column length {weights.shape[1]} does not match workload size {workload_size}.")
        return weights
    return None


def workload_tensor_rows(df: pd.DataFrame, use_norm_wlmix: bool, norm_weights: np.ndarray | None = None) -> list[torch.Tensor]:
    require_columns(df, ["workload_mix"], "workload tensor construction")

    # Parse each unique workload string once. This is much faster than reparsing
    # the same workload profile for every P/R/weight row.
    base_cache: dict[str, np.ndarray] = {}
    for value in pd.unique(df["workload_mix"]):
        base_cache[str(value)] = parse_workload_mix(value)

    first_base = base_cache[str(df["workload_mix"].iloc[0])]
    workload_size = first_base.shape[0]
    weights_matrix = extract_weight_matrix(df, workload_size)

    tensor_cache: dict[tuple[str, tuple[float, ...] | None], torch.Tensor] = {}
    rows = []

    for idx, value in enumerate(df["workload_mix"].to_numpy()):
        key_text = str(value)
        base = base_cache[key_text]
        weights_tuple = None
        if weights_matrix is not None:
            weights_tuple = tuple(float(x) for x in weights_matrix[idx])

        cache_key = (key_text, weights_tuple)
        cached = tensor_cache.get(cache_key)
        if cached is not None:
            rows.append(cached)
            continue

        arr = base.copy()
        if arr.shape[1] == 6:
            if weights_tuple is None:
                raise ValueError("workload_mix has six columns but no Weight_i or weights column was found.")
            weights = np.asarray(weights_tuple, dtype=float)
            arr = np.concatenate([arr, weights.reshape(-1, 1)], axis=1)
        elif arr.shape[1] == 7 and weights_tuple is not None:
            arr[:, 6] = np.asarray(weights_tuple, dtype=float)

        if use_norm_wlmix:
            if norm_weights is None:
                raise ValueError("norm_weights must be provided when use_norm_wlmix=True")
            arr = arr / norm_weights

        tensor = torch.tensor(arr, dtype=torch.float)
        tensor_cache[cache_key] = tensor
        rows.append(tensor)

    return rows


def compute_workload_norm_weights(df: pd.DataFrame) -> np.ndarray:
    # Recompute empirical feature averages from FlexDC rows. This follows the
    # spirit of CONDOR's workload feature normalization while avoiding reuse of
    # constants from a different workload dataset. The weight column is left as 1.
    total = np.zeros(6, dtype=float)
    count = 0
    value_counts = df["workload_mix"].value_counts(dropna=False)
    for value, repeats in value_counts.items():
        arr = parse_workload_mix(value)
        if arr.shape[1] == 7:
            arr = arr[:, :6]
        total += np.sum(np.abs(arr), axis=0) * int(repeats)
        count += arr.shape[0] * int(repeats)
    means = total / max(count, 1)
    means[means == 0] = 1.0
    return np.concatenate([means, np.asarray([1.0])])


class DCDataset(Dataset):
    def __init__(
        self,
        results_csv: str | Path = "../data/initial_data/traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv",
        diagnostics_csv: str | Path | None = None,
        target_family: str = "condor",
        target_mode: str = "normal",
        use_norm_pr: bool = True,
        use_norm_cost: bool | None = None,
        use_norm_wlmix: bool = True,
        pad_wlmix: bool = True,
        raw_qos_aggregation: str = "mean",
    ):
        self.df = read_results_and_diagnostics(results_csv, diagnostics_csv)
        self.len = self.df.shape[0]
        self.target_family = target_family.lower().strip()
        self.target_mode = target_mode.lower().strip()
        self.raw_qos_aggregation = raw_qos_aggregation.lower().strip()

        if use_norm_cost is None:
            # Preserve released CONDOR target scaling for CONDOR labels. FlexDC
            # targets remain in their logged units by default.
            use_norm_cost = self.target_family == "condor"
        self.use_norm_cost = bool(use_norm_cost)

        targets, target_names = build_targets(
            self.df,
            self.target_family,
            self.target_mode,
            self.use_norm_cost,
            self.raw_qos_aggregation,
        )
        self.costs = torch.tensor(targets, dtype=torch.float)
        self.target_names = target_names
        self.target_spec = TargetSpec(
            target_family=self.target_family,
            target_mode=self.target_mode,
            use_norm_cost=self.use_norm_cost,
            raw_qos_aggregation=self.raw_qos_aggregation,
            target_names=target_names,
        )

        require_columns(self.df, ["server_count", "utilization", "workload_mix_size"], "feature construction")
        if use_norm_pr:
            require_columns(self.df, ["Pbar_ratio", "R_ratio"], "normalized P/R input construction")
            p = self.df["Pbar_ratio"].astype(float).to_numpy()
            r = self.df["R_ratio"].astype(float).to_numpy()
        else:
            server_counts = self.df["server_count"].astype(float).to_numpy()
            p = get_p_actual_watts(self.df) / (1000.0 * server_counts)
            r = get_r_actual_watts(self.df) / (1000.0 * server_counts)

        feats = np.zeros((self.len, 5), dtype=float)
        feats[:, 0] = p
        feats[:, 1] = r
        feats[:, 2] = self.df["server_count"].astype(float).to_numpy()
        feats[:, 3] = self.df["utilization"].astype(float).to_numpy()
        feats[:, 4] = self.df["workload_mix_size"].astype(float).to_numpy()
        self.feats = torch.tensor(feats, dtype=torch.float)
        self.feature_names = [
            "Pbar_ratio" if use_norm_pr else "Pbar_kw_per_server",
            "R_ratio" if use_norm_pr else "R_kw_per_server",
            "server_count",
            "utilization",
            "workload_mix_size",
        ]

        self.workload_norm_weights = compute_workload_norm_weights(self.df) if use_norm_wlmix else None
        wl_mix = workload_tensor_rows(self.df, use_norm_wlmix=use_norm_wlmix, norm_weights=self.workload_norm_weights)
        if pad_wlmix:
            wl_mix = pad_sequence(wl_mix, batch_first=True, padding_value=0).float()
        self.wl_mix = wl_mix

    def get_statistics(self):
        print("*** Datacenter Dataset Statistics ***")
        print(f"Rows: {self.len}")
        print(f"Target family/mode: {self.target_family}/{self.target_mode}")
        print(f"Target names: {self.target_names}")
        for idx, name in enumerate(self.target_names):
            values = self.costs[:, idx]
            print(
                f"{name}: mean={torch.mean(values).item():.6g}, "
                f"std={torch.std(values).item():.6g}, "
                f"min={torch.min(values).item():.6g}, max={torch.max(values).item():.6g}"
            )
        if self.workload_norm_weights is not None:
            print("Workload norm weights:", self.workload_norm_weights)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.feats[idx], self.wl_mix[idx], self.costs[idx]


def make_dataset(
    results_csv: str | Path,
    diagnostics_csv: str | Path | None,
    target_family: str,
    target_mode: str,
    use_norm_pr: bool = True,
    use_norm_cost: bool | None = None,
    use_norm_wlmix: bool = True,
    raw_qos_aggregation: str = "mean",
) -> DCDataset:
    return DCDataset(
        results_csv=results_csv,
        diagnostics_csv=diagnostics_csv,
        target_family=target_family,
        target_mode=target_mode,
        use_norm_pr=use_norm_pr,
        use_norm_cost=use_norm_cost,
        use_norm_wlmix=use_norm_wlmix,
        raw_qos_aggregation=raw_qos_aggregation,
    )


def train_model(
    model,
    epochs=150,
    lr=1e-4,
    batch_size=512,
    verbose=False,
    cross_validate=True,
    results_csv="../data/initial_data/traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv",
    diagnostics_csv=None,
    target_family="condor",
    target_mode="normal",
    use_norm_pr=True,
    use_norm_cost=None,
    use_norm_wlmix=True,
    raw_qos_aggregation="mean",
    device_name="auto",
    wandb_run=None,
):
    dc_dataset = DCDataset(
        results_csv=results_csv,
        diagnostics_csv=diagnostics_csv,
        target_family=target_family,
        target_mode=target_mode,
        use_norm_pr=use_norm_pr,
        use_norm_cost=use_norm_cost,
        use_norm_wlmix=use_norm_wlmix,
        raw_qos_aggregation=raw_qos_aggregation,
    )
    start_time = time.time()

    if cross_validate:
        gen = torch.Generator().manual_seed(0)
        train_dc_ds, test_dc_dataset = random_split(dc_dataset, [0.7, 0.3], generator=gen)
        dc_train_dataloader = DataLoader(train_dc_ds, batch_size=batch_size, shuffle=True)
        dc_test_dataloader = DataLoader(test_dc_dataset, batch_size=batch_size, shuffle=False)
    else:
        dc_train_dataloader = DataLoader(dc_dataset, batch_size=batch_size, shuffle=True)
        dc_test_dataloader = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = choose_device(device_name)
    model.to(device)

    train_epoch_loss_record = []
    test_epoch_loss_record = []
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for feat_batch, wl_batch, cost_batch in dc_train_dataloader:
            feat_batch = feat_batch.to(device)
            wl_batch = wl_batch.to(device)
            cost_batch = cost_batch.to(device)
            output = model.forward(feat_batch, wl_batch)
            loss = criterion(output, cost_batch)
            optimizer.zero_grad()
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()

        train_loss = total_train_loss / len(dc_train_dataloader)
        train_epoch_loss_record.append(train_loss)

        test_loss = None
        if cross_validate:
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for feat_batch, wl_batch, cost_batch in dc_test_dataloader:
                    feat_batch = feat_batch.to(device)
                    wl_batch = wl_batch.to(device)
                    cost_batch = cost_batch.to(device)
                    output = model.forward(feat_batch, wl_batch)
                    loss = criterion(output, cost_batch)
                    total_test_loss += loss.item()
            test_loss = total_test_loss / len(dc_test_dataloader)
            test_epoch_loss_record.append(test_loss)

        if wandb_run is not None:
            log_payload = {"epoch": epoch, "train_loss": train_loss}
            if test_loss is not None:
                log_payload["heldout_loss"] = test_loss
            wandb_run.log(log_payload, step=epoch)

        if verbose:
            if cross_validate:
                print("Epoch", epoch, "Train Loss:", train_loss, "Test Loss:", test_loss)
            else:
                print("Epoch", epoch, "Loss:", train_loss)

    running_time = time.time() - start_time
    if verbose:
        time_to_carbon(running_time)
    return model, train_epoch_loss_record, test_epoch_loss_record, dc_dataset.target_names


def predict_arrays(model, dataloader, device: torch.device):
    y_true = []
    y_pred = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for feat_batch, wl_batch, cost_batch in dataloader:
            feat_batch = feat_batch.to(device)
            wl_batch = wl_batch.to(device)
            output = model.forward(feat_batch, wl_batch)
            y_true.append(cost_batch.detach().cpu().numpy())
            y_pred.append(output.detach().cpu().numpy())
    return np.vstack(y_true), np.vstack(y_pred)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str], prefix: str) -> dict:
    metrics = {}
    for idx, name in enumerate(target_names):
        true = y_true[:, idx]
        pred = y_pred[:, idx]
        safe_name = name.replace("/", "_")
        metrics[f"{prefix}/{safe_name}_mae"] = float(mean_absolute_error(true, pred))
        metrics[f"{prefix}/{safe_name}_rmse"] = float(math.sqrt(mean_squared_error(true, pred)))
        metrics[f"{prefix}/{safe_name}_r2"] = float(r2_score(true, pred))

    metrics[f"{prefix}/component_mse"] = float(mean_squared_error(y_true, y_pred))
    try:
        metrics[f"{prefix}/component_mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        metrics[f"{prefix}/component_mape"] = float("nan")

    true_sum = y_true.sum(axis=1)
    pred_sum = y_pred.sum(axis=1)
    metrics[f"{prefix}/target_sum_mse"] = float(mean_squared_error(true_sum, pred_sum))
    metrics[f"{prefix}/target_sum_rmse"] = float(math.sqrt(mean_squared_error(true_sum, pred_sum)))
    metrics[f"{prefix}/target_sum_mae"] = float(mean_absolute_error(true_sum, pred_sum))
    metrics[f"{prefix}/target_sum_r2"] = float(r2_score(true_sum, pred_sum))
    try:
        metrics[f"{prefix}/target_sum_mape"] = float(mean_absolute_percentage_error(true_sum, pred_sum))
    except Exception:
        metrics[f"{prefix}/target_sum_mape"] = float("nan")
    return metrics


def evaluate_model(
    model,
    cross_validate=True,
    batch_size=512,
    results_csv="../data/initial_data/traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv",
    diagnostics_csv=None,
    target_family="condor",
    target_mode="normal",
    use_norm_pr=True,
    use_norm_cost=None,
    use_norm_wlmix=True,
    raw_qos_aggregation="mean",
    device_name="auto",
    print_metrics=True,
):
    dc_dataset = DCDataset(
        results_csv=results_csv,
        diagnostics_csv=diagnostics_csv,
        target_family=target_family,
        target_mode=target_mode,
        use_norm_pr=use_norm_pr,
        use_norm_cost=use_norm_cost,
        use_norm_wlmix=use_norm_wlmix,
        raw_qos_aggregation=raw_qos_aggregation,
    )

    if cross_validate:
        gen = torch.Generator().manual_seed(0)
        train_dc_ds, test_dc_dataset = random_split(dc_dataset, [0.7, 0.3], generator=gen)
        dc_train_dataloader = DataLoader(train_dc_ds, batch_size=batch_size, shuffle=False)
        dc_test_dataloader = DataLoader(test_dc_dataset, batch_size=batch_size, shuffle=False)
    else:
        dc_train_dataloader = DataLoader(dc_dataset, batch_size=batch_size, shuffle=False)
        dc_test_dataloader = None

    device = choose_device(device_name)
    y_true_train, y_pred_train = predict_arrays(model, dc_train_dataloader, device)
    train_metrics = metric_dict(y_true_train, y_pred_train, dc_dataset.target_names, "train")

    if cross_validate:
        y_true_test, y_pred_test = predict_arrays(model, dc_test_dataloader, device)
        test_metrics = metric_dict(y_true_test, y_pred_test, dc_dataset.target_names, "heldout")
    else:
        y_true_test = np.empty((0, 3))
        y_pred_test = np.empty((0, 3))
        test_metrics = {}

    all_metrics = {**train_metrics, **test_metrics}

    if print_metrics:
        print("Target names:", dc_dataset.target_names)
        print("MSE  || Train:", train_metrics["train/component_mse"], "| Test:", test_metrics.get("heldout/component_mse", ""))
        print("MAPE || Train:", train_metrics["train/component_mape"], "| Test:", test_metrics.get("heldout/component_mape", ""))
        print("Target-sum MSE  || Train:", train_metrics["train/target_sum_mse"], "| Test:", test_metrics.get("heldout/target_sum_mse", ""))
        print("Target-sum MAPE || Train:", train_metrics["train/target_sum_mape"], "| Test:", test_metrics.get("heldout/target_sum_mape", ""))
        for name in dc_dataset.target_names:
            safe_name = name.replace("/", "_")
            if cross_validate:
                print(f"{name} | heldout MAE={test_metrics[f'heldout/{safe_name}_mae']:.6g} "
                      f"RMSE={test_metrics[f'heldout/{safe_name}_rmse']:.6g} "
                      f"R2={test_metrics[f'heldout/{safe_name}_r2']:.6g}")

    return y_true_train, y_pred_train, y_true_test, y_pred_test, all_metrics, dc_dataset.target_names


def time_to_carbon(training_time, compute_watts=330):
    metric_tons_co2 = compute_watts * 2.7778e-7 * 0.0007 * training_time
    print("Training Time:", training_time)
    print("Metric Tons of Co2:", metric_tons_co2)
