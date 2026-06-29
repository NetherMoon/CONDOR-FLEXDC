"""Sweep CONDOR inference cost weights and validate candidates with FlexDC.

This wrapper intentionally keeps the single-run orchestrator unchanged. For each
cost-weight vector, it calls am_condor_end_to_end_eval.py, reads the generated
validation summary, and writes one master comparison table.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd


START_LABEL = "Starting configuration"
SELECTED_LABEL = "CONDOR-selected configuration"


DEFAULT_WEIGHT_LIST = [
    (0.05, 0.7, 2.0),
    (0.05, 1.0, 2.0),
    (0.05, 2.0, 2.0),
    (0.05, 5.0, 2.0),
    (0.05, 5.0, 5.0),
    (0.05, 5.0, 10.0),
    (0.05, 10.0, 5.0),
    (0.05, 10.0, 10.0),
    (0.05, 20.0, 10.0),
]


def parse_float_list(value: str, *, expected_len: int | None = None, name: str = "value") -> list[float]:
    try:
        values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Could not parse {name} as comma-separated floats: {value}") from exc
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"{name} must have {expected_len} comma-separated floats. Got {values}")
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return values


def parse_weight_vectors(args: argparse.Namespace) -> list[tuple[float, float, float]]:
    if args.cost_weights_list:
        vectors = []
        for chunk in args.cost_weights_list.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            vectors.append(tuple(parse_float_list(chunk, expected_len=3, name="--cost-weights-list")))
        if not vectors:
            raise ValueError("--cost-weights-list did not contain any valid weight vectors.")
        return vectors

    grid_args = [args.power_weights, args.error_weights, args.qos_weights]
    if any(item is not None for item in grid_args):
        if not all(item is not None for item in grid_args):
            raise ValueError("Use all three of --power-weights, --error-weights, and --qos-weights for grid mode.")
        powers = parse_float_list(args.power_weights, name="--power-weights")
        errors = parse_float_list(args.error_weights, name="--error-weights")
        qoss = parse_float_list(args.qos_weights, name="--qos-weights")
        return [(p, e, q) for p, e, q in itertools.product(powers, errors, qoss)]

    return DEFAULT_WEIGHT_LIST


def sanitize_float(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def weights_tag(weights: Iterable[float]) -> str:
    p, e, q = list(weights)
    return f"pw{sanitize_float(p)}_ew{sanitize_float(e)}_qw{sanitize_float(q)}"


def pct_change(start_value: float, selected_value: float) -> float:
    if pd.isna(start_value) or pd.isna(selected_value):
        return float("nan")
    if abs(float(start_value)) < 1e-12:
        return 0.0 if abs(float(selected_value)) < 1e-12 else float("nan")
    return (float(selected_value) - float(start_value)) / abs(float(start_value)) * 100.0


def safe_float(row: pd.Series, name: str) -> float:
    if name not in row.index:
        return float("nan")
    try:
        return float(row[name])
    except (TypeError, ValueError):
        return float("nan")


def get_row(table: pd.DataFrame, label_prefix: str) -> pd.Series:
    matches = table[table["Configuration"].astype(str).str.startswith(label_prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row starting with {label_prefix!r}; found {len(matches)}")
    return matches.iloc[0]


def run_subprocess(command: list[str], *, cwd: Path) -> int:
    print("\n" + "=" * 90)
    print("Running:")
    print(" ".join(command))
    print("=" * 90)
    completed = subprocess.run(command, cwd=str(cwd))
    return int(completed.returncode)


def read_run_summary(run_dir: Path, cost_weights: tuple[float, float, float], run_index: int) -> dict:
    summary_path = run_dir / "end_to_end_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing validation summary: {summary_path}")

    table = pd.read_csv(summary_path)
    start = get_row(table, START_LABEL)
    selected = get_row(table, SELECTED_LABEL)

    selected_p90 = safe_float(selected, "Ctrack_Epsilon_90th")
    start_p90 = safe_float(start, "Ctrack_Epsilon_90th")
    selected_mean_tracking = safe_float(selected, "MeanAbs_Normalized_Tracking_Error")
    start_mean_tracking = safe_float(start, "MeanAbs_Normalized_Tracking_Error")

    row = {
        "Run_Index": run_index,
        "Cost_Weights": ",".join(str(x) for x in cost_weights),
        "Power_Weight": cost_weights[0],
        "Error_Weight": cost_weights[1],
        "QoS_Weight": cost_weights[2],
        "Run_Directory": str(run_dir),
        "Start_Pbar_kw_per_server": safe_float(start, "Pbar_kw_per_server"),
        "Selected_Pbar_kw_per_server": safe_float(selected, "Pbar_kw_per_server"),
        "Start_R_kw_per_server": safe_float(start, "R_kw_per_server"),
        "Selected_R_kw_per_server": safe_float(selected, "R_kw_per_server"),
        "Start_P_minus_R_kw_per_server": safe_float(start, "P_minus_R_kw_per_server"),
        "Selected_P_minus_R_kw_per_server": safe_float(selected, "P_minus_R_kw_per_server"),
        "Selected_Weights": selected.get("Weights", ""),
        "Start_Predicted_Objective": safe_float(start, "Predicted_Optimization_Objective"),
        "Selected_Predicted_Objective": safe_float(selected, "Predicted_Optimization_Objective"),
        "Start_Actual_Objective": safe_float(start, "Actual_Optimization_Objective"),
        "Selected_Actual_Objective": safe_float(selected, "Actual_Optimization_Objective"),
        "Actual_Objective_Change": safe_float(selected, "Actual_Optimization_Objective") - safe_float(start, "Actual_Optimization_Objective"),
        "Actual_Objective_Change_Percent": pct_change(safe_float(start, "Actual_Optimization_Objective"), safe_float(selected, "Actual_Optimization_Objective")),
        "Start_FlexDC_Full_Objective": safe_float(start, "FlexDC_Full_Objective_Cost"),
        "Selected_FlexDC_Full_Objective": safe_float(selected, "FlexDC_Full_Objective_Cost"),
        "FlexDC_Full_Objective_Change_Percent": pct_change(safe_float(start, "FlexDC_Full_Objective_Cost"), safe_float(selected, "FlexDC_Full_Objective_Cost")),
        "Start_Actual_Scaled_cost_power": safe_float(start, "Actual_Scaled_cost_power"),
        "Selected_Actual_Scaled_cost_power": safe_float(selected, "Actual_Scaled_cost_power"),
        "Start_Actual_Scaled_cost_error": safe_float(start, "Actual_Scaled_cost_error"),
        "Selected_Actual_Scaled_cost_error": safe_float(selected, "Actual_Scaled_cost_error"),
        "Start_Actual_Scaled_cost_qos": safe_float(start, "Actual_Scaled_cost_qos"),
        "Selected_Actual_Scaled_cost_qos": safe_float(selected, "Actual_Scaled_cost_qos"),
        "Start_Actual_Raw_cost_power": safe_float(start, "Actual_Raw_cost_power"),
        "Selected_Actual_Raw_cost_power": safe_float(selected, "Actual_Raw_cost_power"),
        "Start_Actual_Raw_cost_error": safe_float(start, "Actual_Raw_cost_error"),
        "Selected_Actual_Raw_cost_error": safe_float(selected, "Actual_Raw_cost_error"),
        "Start_Actual_Raw_cost_qos": safe_float(start, "Actual_Raw_cost_qos"),
        "Selected_Actual_Raw_cost_qos": safe_float(selected, "Actual_Raw_cost_qos"),
        "Start_MeanAbs_Normalized_Tracking_Error": start_mean_tracking,
        "Selected_MeanAbs_Normalized_Tracking_Error": selected_mean_tracking,
        "Start_Ctrack_Epsilon_90th": start_p90,
        "Selected_Ctrack_Epsilon_90th": selected_p90,
        "Start_QoS_Violation_Ratio": safe_float(start, "QoS_Violation_Ratio"),
        "Selected_QoS_Violation_Ratio": safe_float(selected, "QoS_Violation_Ratio"),
        "Start_QoS_Delay_Probability_Sum": safe_float(start, "QoS_Delay_Probability_Sum"),
        "Selected_QoS_Delay_Probability_Sum": safe_float(selected, "QoS_Delay_Probability_Sum"),
    }
    return row


def add_pass_fail_columns(master: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    master = master.copy()
    master["Pass_QoS"] = master["Selected_QoS_Violation_Ratio"] <= args.max_qos_violation_ratio
    master["Pass_Tracking"] = master["Selected_Ctrack_Epsilon_90th"] <= args.max_p90_tracking_error
    master["Pass_Objective_Improved"] = master["Selected_Actual_Objective"] < master["Start_Actual_Objective"]
    if args.allow_no_objective_improvement:
        master["Pass_All"] = master["Pass_QoS"] & master["Pass_Tracking"]
    else:
        master["Pass_All"] = master["Pass_QoS"] & master["Pass_Tracking"] & master["Pass_Objective_Improved"]

    notes = []
    for _, row in master.iterrows():
        failed = []
        if not bool(row["Pass_QoS"]):
            failed.append("fail_qos")
        if not bool(row["Pass_Tracking"]):
            failed.append("fail_tracking")
        if not bool(row["Pass_Objective_Improved"]):
            failed.append("fail_objective")
        notes.append("pass" if not failed else ";".join(failed))
    master["Status"] = notes

    sort_cols = ["Pass_All", "Pass_QoS", "Pass_Tracking", "Pass_Objective_Improved", "Selected_QoS_Violation_Ratio", "Selected_Ctrack_Epsilon_90th", "Actual_Objective_Change_Percent"]
    ascending = [False, False, False, False, True, True, True]
    master = master.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    master.insert(0, "Rank", np.arange(1, len(master) + 1))
    return master


def write_markdown_report(master: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    columns = [
        "Rank", "Cost_Weights", "Pass_All", "Status",
        "Selected_Actual_Objective", "Actual_Objective_Change_Percent",
        "Selected_Ctrack_Epsilon_90th", "Selected_QoS_Violation_Ratio",
        "Selected_FlexDC_Full_Objective", "Selected_Pbar_kw_per_server", "Selected_R_kw_per_server",
    ]
    report = master[columns].copy()
    for col in report.columns:
        if pd.api.types.is_float_dtype(report[col]):
            report[col] = report[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    text = []
    text.append("# CONDOR Cost-Weight Sweep Summary\n")
    text.append(f"QoS pass threshold: selected QoS violation ratio <= {args.max_qos_violation_ratio}\n")
    text.append(f"Tracking pass threshold: selected p90 tracking error <= {args.max_p90_tracking_error}\n")
    text.append("\nNote: Actual optimization objectives are calculated with each row's own cost weights, so compare pass/fail and operational metrics across rows before comparing objective magnitudes.\n")
    text.append(report.to_markdown(index=False))
    text.append("\n")
    (out_dir / "weight_sweep_report.md").write_text("\n".join(text), encoding="utf-8")


def write_png_table(master: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional display dependency
        print(f"Skipping PNG table because matplotlib is unavailable: {exc}")
        return

    display_cols = [
        "Rank", "Cost_Weights", "Pass_All", "Status",
        "Selected_Actual_Objective", "Actual_Objective_Change_Percent",
        "Selected_Ctrack_Epsilon_90th", "Selected_QoS_Violation_Ratio",
        "Selected_FlexDC_Full_Objective", "Selected_Pbar_kw_per_server", "Selected_R_kw_per_server",
    ]
    data = master[display_cols].head(20).copy()
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(lambda x: "" if pd.isna(x) else f"{x:.5g}")

    fig_height = max(3.0, 0.45 * (len(data) + 2))
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis("off")
    table = ax.table(cellText=data.values, colLabels=data.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#D9EAF7")
        elif "Pass_All" in data.columns and col == data.columns.get_loc("Pass_All"):
            value = str(data.iloc[row - 1, col])
            cell.set_facecolor("#DDEFD8" if value == "True" else "#F7D9D9")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")

    fig.tight_layout()
    fig.savefig(out_dir / "001_weight_sweep_master_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def init_wandb(args: argparse.Namespace, config: dict):
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


def log_wandb_summary(run, master: pd.DataFrame, out_dir: Path) -> None:
    if run is None:
        return
    import wandb
    for idx, row in master.iterrows():
        run.log({
            "sweep/rank": int(row["Rank"]),
            "sweep/pass_all": int(bool(row["Pass_All"])),
            "sweep/selected_actual_objective": row["Selected_Actual_Objective"],
            "sweep/actual_objective_change_percent": row["Actual_Objective_Change_Percent"],
            "sweep/selected_p90_tracking_error": row["Selected_Ctrack_Epsilon_90th"],
            "sweep/selected_qos_violation_ratio": row["Selected_QoS_Violation_Ratio"],
            "sweep/power_weight": row["Power_Weight"],
            "sweep/error_weight": row["Error_Weight"],
            "sweep/qos_weight": row["QoS_Weight"],
        }, step=int(idx))
    run.log({"sweep/master_summary": wandb.Table(dataframe=master)})
    for filename in ["weight_sweep_master_summary.csv", "weight_sweep_ranked_candidates.csv", "weight_sweep_report.md", "001_weight_sweep_master_summary.png"]:
        path = out_dir / filename
        if path.exists():
            run.save(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple am_condor_end_to_end_eval.py evaluations with different CONDOR cost weights.")

    parser.add_argument("--orchestrator-script", default="am_condor_end_to_end_eval.py")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-existing", action="store_true")

    # Cost-weight selection.
    parser.add_argument("--cost-weights-list", default=None, help="Semicolon-separated vectors, e.g. '0.05,0.7,2;0.05,5,2'.")
    parser.add_argument("--power-weights", default=None)
    parser.add_argument("--error-weights", default=None)
    parser.add_argument("--qos-weights", default=None)

    # Pass/fail thresholds.
    parser.add_argument("--max-qos-violation-ratio", type=float, default=0.1)
    parser.add_argument("--max-p90-tracking-error", type=float, default=0.3)
    parser.add_argument("--allow-no-objective-improvement", action="store_true")

    # Arguments forwarded to orchestrator.
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--norm-source-results-csv", required=True)
    parser.add_argument("--start-pbar-kw-per-server", type=float, required=True)
    parser.add_argument("--start-r-kw-per-server", type=float, required=True)
    parser.add_argument("--start-weights", required=True)
    parser.add_argument("--server-count", type=int, default=None)
    parser.add_argument("--utilization", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--flexdc-root", required=True)
    parser.add_argument("--flexdc-python", default=sys.executable)
    parser.add_argument("--gradient-config", default="../../configs/gradient_descent/gradient_descent.ini")
    parser.add_argument("--cluster-config", default="../../configs/cluster/cluster.ini")
    parser.add_argument("--policy-name", default="AQA")
    parser.add_argument("--node-count-control", default="true")
    parser.add_argument("--pbar-lower-factor", type=float, default=0.9)
    parser.add_argument("--pbar-upper-factor", type=float, default=1.0)
    parser.add_argument("--pr-upper-factor", type=float, default=1.2)
    parser.add_argument("--r-lower-kw-per-server", type=float, default=0.01)
    parser.add_argument("--run-flexdc", action="store_true")

    # W&B.
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--no-child-wandb", action="store_true", help="Do not pass W&B arguments to each orchestrator subprocess.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_vectors = parse_weight_vectors(args)

    run = init_wandb(args, {**vars(args), "cost_weight_vectors": weight_vectors})
    rows = []
    failures = []

    for idx, weights in enumerate(weight_vectors, start=1):
        tag = weights_tag(weights)
        run_dir = out_dir / f"run_{idx:03d}_{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "end_to_end_validation_summary.csv"

        if args.skip_existing and summary_path.exists():
            print(f"\nSkipping existing run {idx}: {weights} -> {run_dir}")
        else:
            command = [
                args.python_executable,
                args.orchestrator_script,
                "--model-file", args.model_file,
                "--norm-source-results-csv", args.norm_source_results_csv,
                "--workload-config", args.workload_config,
                "--experiment-config", args.experiment_config,
                "--start-pbar-kw-per-server", str(args.start_pbar_kw_per_server),
                "--start-r-kw-per-server", str(args.start_r_kw_per_server),
                "--start-weights", args.start_weights,
                "--utilization", str(args.utilization),
                "--iterations", str(args.iterations),
                "--lr", str(args.lr),
                "--cost-weights", ",".join(str(x) for x in weights),
                "--device", args.device,
                "--flexdc-root", args.flexdc_root,
                "--flexdc-python", args.flexdc_python,
                "--gradient-config", args.gradient_config,
                "--cluster-config", args.cluster_config,
                "--policy-name", args.policy_name,
                "--node-count-control", args.node_count_control,
                "--pbar-lower-factor", str(args.pbar_lower_factor),
                "--pbar-upper-factor", str(args.pbar_upper_factor),
                "--pr-upper-factor", str(args.pr_upper_factor),
                "--r-lower-kw-per-server", str(args.r_lower_kw_per_server),
                "--out-dir", str(run_dir),
            ]
            if args.server_count is not None:
                command.extend(["--server-count", str(args.server_count)])
            if args.run_flexdc:
                command.append("--run-flexdc")
            if args.wandb_project and not args.no_child_wandb:
                child_name_base = args.wandb_run_name or "weight-sweep"
                command.extend(["--wandb-project", args.wandb_project])
                if args.wandb_entity:
                    command.extend(["--wandb-entity", args.wandb_entity])
                command.extend(["--wandb-run-name", f"{child_name_base}-run{idx:03d}-{tag}"])
                command.extend(["--wandb-mode", args.wandb_mode])

            return_code = run_subprocess(command, cwd=cwd)
            if return_code != 0:
                failures.append({"Run_Index": idx, "Cost_Weights": weights, "Return_Code": return_code, "Run_Directory": str(run_dir)})
                print(f"Run {idx} failed with return code {return_code}; continuing to next weight vector.")
                continue

        try:
            rows.append(read_run_summary(run_dir, weights, idx))
        except Exception as exc:
            failures.append({"Run_Index": idx, "Cost_Weights": weights, "Return_Code": "summary_error", "Error": str(exc), "Run_Directory": str(run_dir)})
            print(f"Could not read summary for run {idx}: {exc}")

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "weight_sweep_failures.csv", index=False)

    if not rows:
        raise RuntimeError("No successful runs were summarized. Check weight_sweep_failures.csv if present.")

    master = pd.DataFrame(rows)
    master = add_pass_fail_columns(master, args)
    master.to_csv(out_dir / "weight_sweep_master_summary.csv", index=False)
    master[master["Pass_All"]].to_csv(out_dir / "weight_sweep_ranked_candidates.csv", index=False)
    write_markdown_report(master, out_dir, args)
    write_png_table(master, out_dir)
    log_wandb_summary(run, master, out_dir)

    print("\nWeight-sweep master summary")
    display_cols = [
        "Rank", "Cost_Weights", "Pass_All", "Status",
        "Selected_Actual_Objective", "Actual_Objective_Change_Percent",
        "Selected_Ctrack_Epsilon_90th", "Selected_QoS_Violation_Ratio",
        "Selected_FlexDC_Full_Objective", "Selected_Pbar_kw_per_server", "Selected_R_kw_per_server",
    ]
    print(master[display_cols].round(6).to_string(index=False))
    print(f"\nSaved: {out_dir / 'weight_sweep_master_summary.csv'}")
    print(f"Saved: {out_dir / 'weight_sweep_ranked_candidates.csv'}")
    print(f"Saved: {out_dir / 'weight_sweep_report.md'}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
