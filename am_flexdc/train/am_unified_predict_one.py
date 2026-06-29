"""Predict one configuration with a unified CONDOR/FlexDC surrogate.

This script keeps the same basic interface as CONDOR's inference helpers:
construct the 5-value simulator feature vector, construct the workload-mix
matrix with weights, load the frozen DataCenterModel, and return its three
predicted target components.

The only extension is that the target semantics are configurable to match the
unified training utility:
    target_family in {condor, flexdc}
    target_mode   in {normal, raw}

FlexDC is not executed by this file. This is model-only prediction.
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
from am_unified_training_utilities import (
    CONDOR_POWER_COST_COEFFICIENT,
    CONDOR_QOS_BETA,
    CONDOR_QOS_RHO,
    CONDOR_QOS_THRESHOLD,
    compute_workload_norm_weights,
)

DEFAULT_CONDOR_EXAMPLE_WEIGHTS = [0.05, 0.7, 2.0]
DEFAULT_FLEXDC_WEIGHTS = [1.0, 1.0, 1.0]


def parse_bool_text(value: str | bool | None, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"auto", ""}:
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{name} must be one of auto,true,false. Got: {value}")


def resolve_use_norm_cost(target_family: str, value: str | bool | None) -> bool:
    parsed = parse_bool_text(value, name="--use-norm-cost")
    if parsed is not None:
        return parsed
    return target_family.lower().strip() == "condor"


def parse_float_list(value: str, expected_len: Optional[int] = None, name: str = "value") -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values; got {len(values)}: {values}")
    return values


def default_objective_weights(target_family: str) -> list[float]:
    if target_family.lower().strip() == "condor":
        return list(DEFAULT_CONDOR_EXAMPLE_WEIGHTS)
    return list(DEFAULT_FLEXDC_WEIGHTS)


def parse_objective_weights(value: str | None, target_family: str) -> list[float]:
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return default_objective_weights(target_family)
    return parse_float_list(value, expected_len=3, name="--objective-weights")


def choose_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device option: {device_name}")


def target_names(target_family: str, target_mode: str, raw_qos_aggregation: str) -> list[str]:
    target_family = target_family.lower().strip()
    target_mode = target_mode.lower().strip()
    raw_qos_aggregation = raw_qos_aggregation.lower().strip()

    if target_family == "condor" and target_mode == "normal":
        return ["condor_cost_power", "condor_cost_error", "condor_cost_qos"]
    if target_family == "condor" and target_mode == "raw":
        return ["condor_cost_power", "condor_cost_error", f"raw_qos_probability_{raw_qos_aggregation}"]
    if target_family == "flexdc" and target_mode == "normal":
        return ["flexdc_M_RSR", "flexdc_Ctrack_weighted", "flexdc_CQoS_weighted"]
    if target_family == "flexdc" and target_mode == "raw":
        return ["flexdc_M_RSR", "raw_Ctrack_Epsilon_90th", f"raw_qos_probability_{raw_qos_aggregation}"]
    raise ValueError("target_family must be condor/flexdc and target_mode must be normal/raw")


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
    # This follows FlexDC JobProfileReader: enumerate parser.sections() in config order.
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
    # Same denominator logic used by the FlexDC data wizard for logging Pbar_ratio/R_ratio.
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


def build_model_inputs(
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    weights: list[float],
    workload_config: str | Path,
    experiment_config: str | Path,
    norm_source_results_csv: str | Path,
    server_count_override=None,
    utilization_override=None,
    use_norm_pr: bool = True,
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

    # Must match am_unified_training_utilities.compute_workload_norm_weights.
    norm_df = pd.read_csv(norm_source_results_csv, usecols=["workload_mix"])
    workload_norm_weights = compute_workload_norm_weights(norm_df)
    workload_7 = np.column_stack([jobs_6, np.asarray(weights, dtype=np.float32)])
    workload_scaled = workload_7 / workload_norm_weights

    if use_norm_pr:
        p_feature = pr["Pbar_ratio"]
        r_feature = pr["R_ratio"]
    else:
        p_feature = float(pbar_kw_per_server)
        r_feature = float(r_kw_per_server)

    sim_features = np.asarray([
        p_feature,
        r_feature,
        exp["server_count"],
        exp["utilization"],
        jobs_6.shape[0],
    ], dtype=np.float32)

    return {
        "sim_features": sim_features,
        "workload_scaled": workload_scaled.astype(np.float32),
        "workload_unscaled": workload_7.astype(np.float32),
        "workload_norm_weights": workload_norm_weights.astype(np.float32),
        "job_names": job_names,
        "server_count": exp["server_count"],
        "utilization": exp["utilization"],
        "idle_watts": exp["idle_watts"],
        "workload_mix_size": jobs_6.shape[0],
        "use_norm_pr": bool(use_norm_pr),
        **pr,
    }


def inverse_scale_targets(
    predicted_targets: np.ndarray,
    target_family: str,
    target_mode: str,
    use_norm_cost: bool,
    raw_qos_aggregation: str,
    server_count: int,
    workload_size: int,
) -> tuple[np.ndarray, list[str]]:
    predicted_targets = np.asarray(predicted_targets, dtype=float)
    names = target_names(target_family, target_mode, raw_qos_aggregation)

    # Only CONDOR targets are scaled by the released CONDOR loader. FlexDC targets
    # are logged/trained in their native units in am_unified_training_utilities.
    if target_family.lower().strip() != "condor" or not use_norm_cost:
        return predicted_targets.copy(), names

    raw = np.asarray([
        predicted_targets[0] * float(server_count) / 120.0,
        predicted_targets[1] * float(server_count) / 200.0,
        predicted_targets[2] * float(workload_size),
    ])

    if target_mode.lower().strip() == "raw" and raw_qos_aggregation.lower().strip() == "mean":
        raw[2] = predicted_targets[2]
    return raw, names


def weighted_objective(prediction_targets: np.ndarray, objective_weights: list[float]) -> float:
    return float(np.dot(np.asarray(objective_weights, dtype=float), np.asarray(prediction_targets, dtype=float)))


def predict_costs(
    model_file: str | Path,
    pbar_kw_per_server: float,
    r_kw_per_server: float,
    weights: list[float],
    workload_config: str | Path,
    experiment_config: str | Path,
    norm_source_results_csv: str | Path,
    target_family: str = "condor",
    target_mode: str = "normal",
    raw_qos_aggregation: str = "mean",
    use_norm_cost: bool | None = None,
    use_norm_pr: bool = True,
    objective_weights: list[float] | None = None,
    device_name: str = "auto",
    server_count_override=None,
    utilization_override=None,
) -> dict:
    target_family = target_family.lower().strip()
    target_mode = target_mode.lower().strip()
    raw_qos_aggregation = raw_qos_aggregation.lower().strip()
    use_norm_cost = resolve_use_norm_cost(target_family, use_norm_cost)
    if objective_weights is None:
        objective_weights = default_objective_weights(target_family)

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
        use_norm_pr=use_norm_pr,
    )

    model = DataCenterModel()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        sim = torch.tensor(inputs["sim_features"], dtype=torch.float32, device=device)
        workload = torch.tensor(inputs["workload_scaled"], dtype=torch.float32, device=device).unsqueeze(0)
        pred_targets = model(sim, workload).detach().cpu().numpy().reshape(-1)

    unscaled, names = inverse_scale_targets(
        pred_targets,
        target_family,
        target_mode,
        use_norm_cost,
        raw_qos_aggregation,
        inputs["server_count"],
        inputs["workload_mix_size"],
    )

    result = {
        "device": str(device),
        "target_family": target_family,
        "target_mode": target_mode,
        "use_norm_cost": bool(use_norm_cost),
        "use_norm_pr": bool(use_norm_pr),
        "raw_qos_aggregation": raw_qos_aggregation,
        "target_names": names,
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
        "Predicted_Optimization_Objective": weighted_objective(pred_targets, objective_weights),
        "Predicted_Target_Sum": float(np.sum(pred_targets)),
        "objective_weights": [float(x) for x in objective_weights],
        "workload_norm_weights": inputs["workload_norm_weights"].astype(float).tolist(),
        "job_names": inputs["job_names"],
    }
    for idx, name in enumerate(names):
        result[f"Predicted_{name}"] = float(pred_targets[idx])
        result[f"Predicted_unscaled_{name}"] = float(unscaled[idx])
    return result


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
    parser = argparse.ArgumentParser(description="Predict one configuration with a unified CONDOR/FlexDC surrogate.")
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--norm-source-results-csv", required=True)
    parser.add_argument("--target-family", choices=["condor", "flexdc"], required=True)
    parser.add_argument("--target-mode", choices=["normal", "raw"], required=True)
    parser.add_argument("--raw-qos-aggregation", choices=["mean", "sum"], default="mean")
    parser.add_argument("--use-norm-cost", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--use-norm-pr", choices=["true", "false"], default="true")
    parser.add_argument("--pbar-kw-per-server", type=float, required=True)
    parser.add_argument("--r-kw-per-server", type=float, required=True)
    parser.add_argument("--weights", required=True, help="Comma-separated workload weights, one per job type.")
    parser.add_argument("--server-count", type=int, default=None, help="Optional override for experiment config server_count.")
    parser.add_argument("--utilization", type=float, default=None, help="Optional override for experiment config utilization.")
    parser.add_argument("--objective-weights", default="auto", help="Comma-separated weights for the three model outputs, or auto.")
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
    use_norm_cost = resolve_use_norm_cost(args.target_family, args.use_norm_cost)
    use_norm_pr = parse_bool_text(args.use_norm_pr, name="--use-norm-pr")
    objective_weights = parse_objective_weights(args.objective_weights, args.target_family)

    result = predict_costs(
        model_file=args.model_file,
        pbar_kw_per_server=args.pbar_kw_per_server,
        r_kw_per_server=args.r_kw_per_server,
        weights=weights,
        workload_config=args.workload_config,
        experiment_config=args.experiment_config,
        norm_source_results_csv=args.norm_source_results_csv,
        target_family=args.target_family,
        target_mode=args.target_mode,
        raw_qos_aggregation=args.raw_qos_aggregation,
        use_norm_cost=use_norm_cost,
        use_norm_pr=bool(use_norm_pr),
        objective_weights=objective_weights,
        device_name=args.device,
        server_count_override=args.server_count,
        utilization_override=args.utilization,
    )

    run = init_wandb(args, {**vars(args), "objective_weights": objective_weights, "use_norm_cost_resolved": use_norm_cost})
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
