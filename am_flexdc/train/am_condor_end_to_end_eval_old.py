"""Orchestrate CONDOR-style optimization and FlexDC validation.

This script runs the frozen CONDOR/FlexDC model to select P,R,w, then runs the
FlexDC data extraction wizard for the starting and selected configurations, and
finally compares predicted CONDOR-style costs with actual FlexDC-derived costs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from am_condor_optimize_one import optimize_inputs
from am_condor_predict_one import (
    CONDOR_POWER_COST_COEFFICIENT,
    CONDOR_QOS_BETA,
    CONDOR_QOS_RHO,
    CONDOR_QOS_THRESHOLD,
    DEFAULT_COST_WEIGHTS,
    parse_float_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize with CONDOR model, validate in FlexDC, compare results.")

    # Model/inference arguments.
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--norm-source-results-csv", required=True)
    parser.add_argument("--start-pbar-kw-per-server", type=float, required=True)
    parser.add_argument("--start-r-kw-per-server", type=float, required=True)
    parser.add_argument("--start-weights", required=True)
    parser.add_argument("--server-count", type=int, default=None)
    parser.add_argument("--utilization", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--cost-weights", default=",".join(str(x) for x in DEFAULT_COST_WEIGHTS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", default="condor_end_to_end_eval")

    # Constraints used by optimizer and FlexDC manual validation.
    parser.add_argument("--pbar-lower-factor", type=float, default=0.9)
    parser.add_argument("--pbar-upper-factor", type=float, default=1.0)
    parser.add_argument("--pr-upper-factor", type=float, default=1.2)
    parser.add_argument("--r-lower-kw-per-server", type=float, default=0.01)

    # FlexDC execution arguments.
    parser.add_argument("--flexdc-root", required=True, help="Path to FlexDC repository root.")
    parser.add_argument("--flexdc-python", default=sys.executable, help="Python executable used to run am_data_extraction_wizard.py.")
    parser.add_argument("--gradient-config", default="../../configs/gradient_descent/gradient_descent.ini")
    parser.add_argument("--cluster-config", default="../../configs/cluster/cluster.ini")
    parser.add_argument("--policy-name", default="AQA")
    parser.add_argument("--node-count-control", default="true")
    parser.add_argument("--run-flexdc", action="store_true", help="Actually run FlexDC validation. Omit for optimization-only dry run.")

    # W&B is optional. If --wandb-project is omitted, no W&B logging occurs.
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


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


def path_for_wizard(path_text: str) -> str:
    """Return a path safe to pass to FlexDC after changing cwd to src/peacsim.

    If the path exists relative to the current orchestrator working directory,
    pass an absolute path. Otherwise keep it unchanged so FlexDC-native defaults
    like ../../configs/... remain relative to src/peacsim.
    """
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    return path_text


def run_wizard(args: argparse.Namespace, label: str, pbar: float, reserve: float, weights: list[float]) -> tuple[Path, Path]:
    flexdc_root = Path(args.flexdc_root).resolve()
    peacsim_dir = flexdc_root / "src" / "peacsim"
    output_label = f"{Path(args.out_dir).name}_{label}"
    command = [
        args.flexdc_python,
        "-u",
        "am_data_extraction_wizard.py",
        "--gradient-config", path_for_wizard(args.gradient_config),
        "--experiment-config", path_for_wizard(args.experiment_config),
        "--cluster-config", path_for_wizard(args.cluster_config),
        "--policy-name", args.policy_name,
        "--job-config", path_for_wizard(args.workload_config),
        "--output-dir", output_label,
        "--utilization-values", str(args.utilization),
        "--auto-workload-pr-sweep", "false",
        "--pbar-kw-per-server-values", str(pbar),
        "--r-kw-per-server-values", str(reserve),
        "--weight-vectors", ",".join(str(x) for x in weights),
        "--node-count-control", args.node_count_control,
        "--pbar-lower-factor", str(args.pbar_lower_factor),
        "--pbar-upper-factor", str(args.pbar_upper_factor),
        "--pr-upper-factor", str(args.pr_upper_factor),
        "--pr-chunk-index", "0",
        "--pr-num-chunks", "1",
    ]
    print("\nRunning FlexDC", label)
    print(" ".join(command))
    env = os.environ.copy()
    src_path = str(flexdc_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=str(peacsim_dir), env=env, check=True)

    opt_root = peacsim_dir / "output" / "optimization"
    folders = sorted(opt_root.glob(output_label + "_*"), key=lambda p: p.stat().st_mtime)
    if not folders:
        raise FileNotFoundError(f"Could not find FlexDC output folder for {output_label}")
    folder = folders[-1]
    return folder / "grid_search_results.csv", folder / "grid_search_diagnostics.csv"


def labels_from_results_csv(path: Path) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    if len(data) != 1:
        raise ValueError(f"Expected one FlexDC result row in {path}, found {len(data)}")
    row = data.iloc[0]
    n = float(row["server_count"])
    j = float(row["workload_mix_size"])
    raw_power = CONDOR_POWER_COST_COEFFICIENT * (float(row["P_actual_watts"]) - float(row["R_actual_watts"]))
    raw_error = float(row["Mtrack_Error_MeanAbs_Watts"]) / 1000.0
    probabilities = np.asarray(json.loads(row["QoS_Delay_Probabilities"]), dtype=float)
    raw_qos = CONDOR_QOS_BETA * np.logaddexp(0, CONDOR_QOS_RHO * (probabilities - CONDOR_QOS_THRESHOLD)).sum()
    scaled = np.asarray([raw_power * 120.0 / n, raw_error * 200.0 / n, raw_qos / j])
    return row, np.asarray([raw_power, raw_error, raw_qos]), scaled


def make_validation_table(cost_weights: list[float], prediction_table: pd.DataFrame, start_results: Path, opt_results: Path) -> pd.DataFrame:
    weights = np.asarray(cost_weights, dtype=float)
    pred = prediction_table.set_index("Configuration")
    rows = []
    for label, path in [("Starting configuration", start_results), ("CONDOR-selected configuration", opt_results)]:
        source, raw, scaled = labels_from_results_csv(path)
        predicted = pred.loc[label]
        weight_cols = sorted([col for col in source.index if col.startswith("Weight_") and col != "Weight_Sample_ID"], key=lambda name: int(name.split("_")[-1]))
        rows.append({
            "Configuration": label,
            "Pbar_kw_per_server": float(source["Pbar_kw_per_server"]),
            "R_kw_per_server": float(source["R_kw_per_server"]),
            "Weights": json.dumps([float(source[col]) for col in weight_cols]),
            "Predicted_Optimization_Objective": float(predicted["Predicted_Optimization_Objective"]),
            "Actual_Optimization_Objective": float(np.dot(weights, scaled)),
            "Predicted_Scaled_cost_power": float(predicted["Predicted_Scaled_cost_power"]),
            "Actual_Scaled_cost_power": scaled[0],
            "Predicted_Scaled_cost_error": float(predicted["Predicted_Scaled_cost_error"]),
            "Actual_Scaled_cost_error": scaled[1],
            "Predicted_Scaled_cost_qos": float(predicted["Predicted_Scaled_cost_qos"]),
            "Actual_Scaled_cost_qos": scaled[2],
            "Actual_Raw_cost_power": raw[0],
            "Actual_Raw_cost_error": raw[1],
            "Actual_Raw_cost_qos": raw[2],
            "MeanAbs_Normalized_Tracking_Error": float(source["Mtrack_Error_MeanAbs_Normalized"]),
            "QoS_Violation_Ratio": float(source["QoS_Violation_Ratio"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_weights = parse_float_list(args.start_weights, name="--start-weights")
    cost_weights = parse_float_list(args.cost_weights, expected_len=3, name="--cost-weights")

    if args.run_flexdc and args.utilization is None:
        raise ValueError("--utilization is required when --run-flexdc is used so the validation run matches inference.")

    run = init_wandb(args, {**vars(args), "cost_weights": cost_weights})

    trajectory, candidate, prediction_table = optimize_inputs(
        args.model_file,
        args.workload_config,
        args.experiment_config,
        args.norm_source_results_csv,
        args.start_pbar_kw_per_server,
        args.start_r_kw_per_server,
        start_weights,
        cost_weights=cost_weights,
        iterations=args.iterations,
        lr=args.lr,
        device_name=args.device,
        server_count_override=args.server_count,
        utilization_override=args.utilization,
        pbar_lower_factor=args.pbar_lower_factor,
        pbar_upper_factor=args.pbar_upper_factor,
        pr_upper_factor=args.pr_upper_factor,
        r_lower=args.r_lower_kw_per_server,
    )
    trajectory.to_csv(out_dir / "optimization_trajectory.csv", index=False)
    prediction_table.to_csv(out_dir / "optimization_comparison_before_validation.csv", index=False)
    with open(out_dir / "optimized_candidate.json", "w") as f:
        json.dump(candidate, f, indent=2)

    if run is not None:
        for _, row in trajectory.iterrows():
            run.log({
                "optimization/objective": row["Predicted_Optimization_Objective"],
                "optimization/pbar_kw_per_server": row["Pbar_kw_per_server"],
                "optimization/r_kw_per_server": row["R_kw_per_server"],
            }, step=int(row["Iteration"]))

    if not args.run_flexdc:
        print("\nOptimization complete. Add --run-flexdc to execute FlexDC validation.")
        print(prediction_table[["Configuration", "Pbar_kw_per_server", "R_kw_per_server", "Predicted_Optimization_Objective"]].to_string(index=False))
        if run is not None:
            run.finish()
        return

    start_results, _ = run_wizard(args, "start", candidate["starting_pbar_kw_per_server"], candidate["starting_r_kw_per_server"], candidate["starting_weights"])
    opt_results, _ = run_wizard(args, "optimized", candidate["optimized_pbar_kw_per_server"], candidate["optimized_r_kw_per_server"], candidate["optimized_weights"])
    table = make_validation_table(cost_weights, prediction_table, start_results, opt_results)
    table.to_csv(out_dir / "end_to_end_validation_summary.csv", index=False)

    print("\nEnd-to-end validation summary")
    print(table.round(6).to_string(index=False))
    print(f"\nSaved: {out_dir / 'end_to_end_validation_summary.csv'}")

    if run is not None:
        for _, row in table.iterrows():
            prefix = "start" if row["Configuration"].startswith("Starting") else "selected"
            run.log({
                f"validation/{prefix}_predicted_objective": row["Predicted_Optimization_Objective"],
                f"validation/{prefix}_actual_objective": row["Actual_Optimization_Objective"],
                f"validation/{prefix}_tracking_error": row["MeanAbs_Normalized_Tracking_Error"],
                f"validation/{prefix}_qos_violation_ratio": row["QoS_Violation_Ratio"],
            })
        run.finish()


if __name__ == "__main__":
    main()
