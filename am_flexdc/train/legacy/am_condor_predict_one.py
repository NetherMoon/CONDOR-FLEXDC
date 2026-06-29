"""Predict CONDOR-style costs for one FlexDC configuration.

This script is intentionally small and follows the released CONDOR interface:
DataCenterModel takes five tabular inputs plus a workload-mix matrix and returns
three scaled cost components. FlexDC is used only to supply workload/config
features and later validation data.

Model inputs used here:
    [Pbar_ratio, R_ratio, server_count, utilization, workload_mix_size]
    workload rows [pmin, pmax, Tmin, Tmax, qos, job_size, weight]

Model outputs are the scaled targets used by the released CONDOR loader:
    y_power = 120 * cost_power / server_count
    y_error = 200 * cost_error / server_count
    y_qos   = cost_qos / workload_mix_size
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from data_center_model import DataCenterModel


CONDOR_POWER_COST_COEFFICIENT = 3e-4
CONDOR_QOS_BETA = 0.8
CONDOR_QOS_RHO = 60.0
CONDOR_QOS_THRESHOLD = 0.1
DEFAULT_COST_WEIGHTS = [0.05, 0.7, 2.0]


def choose_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_float_list(value: str, expected_len: Optional[int] = None, name: str = "value") -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values; got {len(values)}: {values}")
    return values


def read_experiment_config(path: str | Path, server_count_override=None, utilization_override=None) -> dict:
    parser = configparser.ConfigParser()
    parser.read(path)
    if "system" not in parser:
        raise ValueError(f"Experiment config has no [system] section: {path}")
    system = parser["system"]
    return {
        "server_count": int(server_count_override) if server_count_override is not None else system.getint("server_count"),
        "utilization": float(utilization_override) if utilization_override is not None else system.getfloat("utilization"),
        "idle_watts": system.getfloat("idle_watts"),
    }


def read_workload_config(path: str | Path) -> tuple[np.ndarray, list[str]]:
    parser = configparser.ConfigParser()
    parser.read(path)
    default_job_size = parser.defaults().get("job_size", "1")

    rows = []
    names = []
    for section in parser.sections():
        job = parser[section]
        rows.append([
            job.getfloat("min_job_power_watts"),
            job.getfloat("max_job_power_watts"),
            job.getfloat("min_time_seconds"),
            job.getfloat("max_time_seconds"),
            job.getfloat("qos_constraint"),
            float(job.get("job_size", default_job_size)),
        ])
        names.append(section)

    if not rows:
        raise ValueError(f"No workload sections found in {path}")
    return np.asarray(rows, dtype=np.float32), names


def calculate_pr_features(
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    server_count: int,
    utilization: float,
    idle_watts: float,
    workload_jobs_6: np.ndarray,
) -> dict:
    avg_max_job_power = float(workload_jobs_6[:, 1].mean())
    avg_num_servers = float(server_count) * float(utilization)
    avg_idle_servers = float(server_count) - avg_num_servers

    pbar_denominator_watts = avg_max_job_power * avg_num_servers + idle_watts * avg_idle_servers
    r_denominator_watts = (avg_max_job_power * server_count - idle_watts * server_count) / 2.0
    if pbar_denominator_watts <= 0 or r_denominator_watts <= 0:
        raise ValueError("Invalid P/R denominators. Check workload max powers and idle_watts.")

    p_actual_watts = float(pbar_kw_per_server) * 1000.0 * float(server_count)
    r_actual_watts = float(r_kw_per_server) * 1000.0 * float(server_count)

    return {
        "P_actual_watts": p_actual_watts,
        "R_actual_watts": r_actual_watts,
        "Pbar_ratio": p_actual_watts / pbar_denominator_watts,
        "R_ratio": r_actual_watts / r_denominator_watts,
        "Pbar_denominator_watts": pbar_denominator_watts,
        "R_denominator_watts": r_denominator_watts,
    }


def compute_workload_norm_weights(results_csv: str | Path) -> np.ndarray:
    """Recompute the workload-feature normalizer used by the CONDOR/FlexDC loader.

    The CONDOR paper normalizes workload features by empirical feature averages.
    The released code hard-coded those values for the original dataset. For the
    FlexDC-generated data, the training utility recomputes them from the training
    CSV, so inference must do the same or load equivalent saved values.
    """
    data = pd.read_csv(results_csv, usecols=["workload_mix"])

    # The first six workload features repeat many times. Compute their empirical
    # average by unique workload profile instead of reparsing every row. The
    # seventh feature is the workload weight, and the released CONDOR loader keeps
    # that normalizer fixed at 1.0.
    total = np.zeros(6, dtype=float)
    count = 0
    for workload_text, repetitions in data["workload_mix"].value_counts().items():
        jobs = np.asarray(json.loads(workload_text), dtype=float)
        if jobs.ndim != 2 or jobs.shape[1] != 6:
            raise ValueError("Expected each workload_mix row to have shape J x 6 in the FlexDC results CSV.")
        total += jobs.sum(axis=0) * int(repetitions)
        count += jobs.shape[0] * int(repetitions)

    if count <= 0:
        raise ValueError("Could not compute workload normalization weights from an empty CSV.")
    norm_weights = np.concatenate([total / count, np.asarray([1.0])])
    norm_weights[norm_weights == 0] = 1.0
    return norm_weights.astype(np.float32)


def build_model_inputs(
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    weights: list[float],
    workload_config: str | Path,
    experiment_config: str | Path,
    norm_source_results_csv: str | Path,
    server_count_override=None,
    utilization_override=None,
) -> dict:
    jobs_6, job_names = read_workload_config(workload_config)
    if len(weights) != jobs_6.shape[0]:
        raise ValueError(f"Expected {jobs_6.shape[0]} weights for workload, got {len(weights)}.")
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(f"Workload weights must sum to 1. Got {sum(weights)}")

    exp = read_experiment_config(experiment_config, server_count_override, utilization_override)
    pr = calculate_pr_features(
        pbar_kw_per_server,
        r_kw_per_server,
        exp["server_count"],
        exp["utilization"],
        exp["idle_watts"],
        jobs_6,
    )

    workload_norm_weights = compute_workload_norm_weights(norm_source_results_csv)
    workload_7 = np.column_stack([jobs_6, np.asarray(weights, dtype=np.float32)])
    workload_scaled = workload_7 / workload_norm_weights
    sim_features = np.asarray([
        pr["Pbar_ratio"],
        pr["R_ratio"],
        exp["server_count"],
        exp["utilization"],
        jobs_6.shape[0],
    ], dtype=np.float32)

    return {
        "sim_features": sim_features,
        "workload_scaled": workload_scaled.astype(np.float32),
        "workload_unscaled": workload_7.astype(np.float32),
        "workload_norm_weights": workload_norm_weights,
        "job_names": job_names,
        "server_count": exp["server_count"],
        "utilization": exp["utilization"],
        "idle_watts": exp["idle_watts"],
        "workload_mix_size": jobs_6.shape[0],
        **pr,
    }


def inverse_scale_predictions(prediction_scaled: np.ndarray, server_count: int, workload_size: int) -> np.ndarray:
    prediction_scaled = np.asarray(prediction_scaled, dtype=float)
    return np.asarray([
        prediction_scaled[0] * float(server_count) / 120.0,
        prediction_scaled[1] * float(server_count) / 200.0,
        prediction_scaled[2] * float(workload_size),
    ])


def weighted_objective(prediction_scaled: np.ndarray, cost_weights: list[float]) -> float:
    return float(np.dot(np.asarray(cost_weights, dtype=float), np.asarray(prediction_scaled, dtype=float)))


def predict_costs(
    model_file: str | Path,
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    weights: list[float],
    workload_config: str | Path,
    experiment_config: str | Path,
    norm_source_results_csv: str | Path,
    cost_weights: list[float] = DEFAULT_COST_WEIGHTS,
    device_name: str = "auto",
    server_count_override=None,
    utilization_override=None,
) -> dict:
    device = choose_device(device_name)
    inputs = build_model_inputs(
        pbar_kw_per_server,
        r_kw_per_server,
        weights,
        workload_config,
        experiment_config,
        norm_source_results_csv,
        server_count_override=server_count_override,
        utilization_override=utilization_override,
    )

    model = DataCenterModel()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        sim = torch.tensor(inputs["sim_features"], dtype=torch.float32, device=device)
        workload = torch.tensor(inputs["workload_scaled"], dtype=torch.float32, device=device).unsqueeze(0)
        scaled = model(sim, workload).detach().cpu().numpy().reshape(-1)

    raw = inverse_scale_predictions(scaled, inputs["server_count"], inputs["workload_mix_size"])
    return {
        "device": str(device),
        "Pbar_kw_per_server": float(pbar_kw_per_server),
        "R_kw_per_server": float(r_kw_per_server),
        "Weights": [float(x) for x in weights],
        "P_actual_watts": inputs["P_actual_watts"],
        "R_actual_watts": inputs["R_actual_watts"],
        "Pbar_ratio": inputs["Pbar_ratio"],
        "R_ratio": inputs["R_ratio"],
        "server_count": inputs["server_count"],
        "utilization": inputs["utilization"],
        "workload_mix_size": inputs["workload_mix_size"],
        "Predicted_Scaled_cost_power": float(scaled[0]),
        "Predicted_Scaled_cost_error": float(scaled[1]),
        "Predicted_Scaled_cost_qos": float(scaled[2]),
        "Predicted_Raw_cost_power": float(raw[0]),
        "Predicted_Raw_cost_error": float(raw[1]),
        "Predicted_Raw_cost_qos": float(raw[2]),
        "Predicted_Optimization_Objective": weighted_objective(scaled, cost_weights),
        "cost_weights": [float(x) for x in cost_weights],
        "workload_norm_weights": inputs["workload_norm_weights"].astype(float).tolist(),
        "job_names": inputs["job_names"],
    }


def init_wandb(args, config: dict):
    if not args.wandb_project:
        return None
    import wandb
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CONDOR-style model costs for one FlexDC configuration.")
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--norm-source-results-csv", required=True)
    parser.add_argument("--pbar-kw-per-server", type=float, required=True)
    parser.add_argument("--r-kw-per-server", type=float, required=True)
    parser.add_argument("--weights", required=True, help="Comma-separated workload weights, one per job type.")
    parser.add_argument("--server-count", type=int, default=None, help="Optional override for experiment config server_count.")
    parser.add_argument("--utilization", type=float, default=None, help="Optional override for experiment config utilization.")
    parser.add_argument("--cost-weights", default=",".join(str(x) for x in DEFAULT_COST_WEIGHTS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = parse_float_list(args.weights, name="--weights")
    cost_weights = parse_float_list(args.cost_weights, expected_len=3, name="--cost-weights")

    result = predict_costs(
        model_file=args.model_file,
        pbar_kw_per_server=args.pbar_kw_per_server,
        r_kw_per_server=args.r_kw_per_server,
        weights=weights,
        workload_config=args.workload_config,
        experiment_config=args.experiment_config,
        norm_source_results_csv=args.norm_source_results_csv,
        cost_weights=cost_weights,
        device_name=args.device,
        server_count_override=args.server_count,
        utilization_override=args.utilization,
    )

    run = init_wandb(args, {**vars(args), "cost_weights": cost_weights})
    if run is not None:
        run.log({k: v for k, v in result.items() if isinstance(v, (int, float))})

    print(json.dumps(result, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(args.out_csv, index=False)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
