"""Run unified surrogate optimization and optional FlexDC validation.

This is the end-to-end wrapper for the four model variants trained by
am_unified_training_utilities.py. It keeps the same role as the previous
single-family orchestrator:
    1. optimize P, R, w with the frozen surrogate;
    2. optionally run FlexDC's data extraction wizard for start/selected points;
    3. reconstruct the same target labels from FlexDC output;
    4. compare predicted vs actual values in a small table.
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

from am_unified_optimize_one import optimize_inputs
from am_unified_predict_one import (
    default_objective_weights,
    parse_bool_text,
    parse_float_list,
    parse_objective_weights,
    resolve_use_norm_cost,
    target_names,
)
from am_unified_training_utilities import build_targets, read_results_and_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize with unified surrogate, validate in FlexDC, compare results.")

    # Model/inference arguments.
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--norm-source-results-csv", required=True)
    parser.add_argument("--target-family", choices=["condor", "flexdc"], required=True)
    parser.add_argument("--target-mode", choices=["normal", "raw"], required=True)
    parser.add_argument("--raw-qos-aggregation", choices=["mean", "sum"], default="mean")
    parser.add_argument("--use-norm-cost", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--use-norm-pr", choices=["true", "false"], default="true")
    parser.add_argument("--start-pbar-kw-per-server", type=float, required=True)
    parser.add_argument("--start-r-kw-per-server", type=float, required=True)
    parser.add_argument("--start-weights", required=True)
    parser.add_argument("--server-count", type=int, default=None)
    parser.add_argument("--utilization", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--objective-weights", default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", default="unified_end_to_end_eval")

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
    """Return a path safe to pass to FlexDC after changing cwd to src/peacsim."""
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


def actual_targets_from_flexdc(
    results_csv: Path,
    diagnostics_csv: Path,
    target_family: str,
    target_mode: str,
    use_norm_cost: bool,
    raw_qos_aggregation: str,
) -> tuple[pd.Series, np.ndarray, list[str]]:
    df = read_results_and_diagnostics(results_csv, diagnostics_csv)
    if len(df) != 1:
        raise ValueError(f"Expected one FlexDC result row in {results_csv}; found {len(df)}")
    targets, names = build_targets(df, target_family, target_mode, use_norm_cost, raw_qos_aggregation)
    return df.iloc[0], targets[0].astype(float), names


def pct_change(start: float, selected: float) -> float:
    if abs(start) < 1e-12:
        return float("nan")
    return float((selected - start) / abs(start) * 100.0)


def make_validation_table(
    objective_weights: list[float],
    prediction_table: pd.DataFrame,
    start_results: Path,
    start_diagnostics: Path,
    opt_results: Path,
    opt_diagnostics: Path,
    target_family: str,
    target_mode: str,
    use_norm_cost: bool,
    raw_qos_aggregation: str,
) -> pd.DataFrame:
    weights = np.asarray(objective_weights, dtype=float)
    pred = prediction_table.set_index("Configuration")
    names = target_names(target_family, target_mode, raw_qos_aggregation)
    rows = []
    for label, result_path, diag_path, pred_label in [
        ("Starting configuration", start_results, start_diagnostics, "Starting configuration"),
        ("Selected configuration", opt_results, opt_diagnostics, "Selected configuration"),
    ]:
        source, actual, actual_names = actual_targets_from_flexdc(
            result_path, diag_path, target_family, target_mode, use_norm_cost, raw_qos_aggregation
        )
        if actual_names != names:
            raise ValueError(f"Target-name mismatch: predicted={names}, actual={actual_names}")
        predicted = pred.loc[pred_label]
        weight_cols = sorted(
            [col for col in source.index if str(col).startswith("Weight_") and str(col) != "Weight_Sample_ID"],
            key=lambda name: int(str(name).split("_")[-1]),
        )
        row = {
            "Configuration": label,
            "Pbar_kw_per_server": float(source["Pbar_kw_per_server"]),
            "R_kw_per_server": float(source["R_kw_per_server"]),
            "Weights": json.dumps([float(source[col]) for col in weight_cols]),
            "Predicted_Optimization_Objective": float(predicted["Predicted_Optimization_Objective"]),
            "Actual_Optimization_Objective": float(np.dot(weights, actual)),
            "Predicted_Target_Sum": float(predicted["Predicted_Target_Sum"]),
            "Actual_Target_Sum": float(np.sum(actual)),
        }
        for idx, name in enumerate(names):
            row[f"Predicted_{name}"] = float(predicted[f"Predicted_{name}"])
            row[f"Actual_{name}"] = float(actual[idx])
        for col in [
            "Simulator_RSR_Total_Cost",
            "Simulator_Power_Cost",
            "Mtrack_Cost",
            "Ctrack_Epsilon_90th",
            "Ctrack_Weighted_Cost",
            "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
            "Diagnostic_FullPaperObjective_Cost",
            "Mtrack_Error_MeanAbs_Normalized",
            "QoS_Delay_Probability_Sum",
            "QoS_Violation_Ratio",
        ]:
            row[col] = float(source[col]) if col in source.index and not pd.isna(source[col]) else np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    start_obj = float(table.loc[0, "Actual_Optimization_Objective"])
    sel_obj = float(table.loc[1, "Actual_Optimization_Objective"])
    table["Actual_Objective_Change_vs_Start"] = [0.0, sel_obj - start_obj]
    table["Actual_Objective_Change_Percent_vs_Start"] = [0.0, pct_change(start_obj, sel_obj)]
    return table


def write_markdown_report(table: pd.DataFrame, out_path: Path, target_family: str, target_mode: str, objective_weights: list[float]) -> None:
    cols = [
        "Configuration",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "Predicted_Optimization_Objective",
        "Actual_Optimization_Objective",
        "Actual_Objective_Change_Percent_vs_Start",
        "Ctrack_Epsilon_90th",
        "QoS_Violation_Ratio",
        "Diagnostic_FullPaperObjective_Cost",
    ]
    available = [c for c in cols if c in table.columns]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Unified end-to-end validation\n\n")
        f.write(f"Target family/mode: `{target_family}/{target_mode}`\n\n")
        f.write(f"Objective weights: `{objective_weights}`\n\n")
        f.write(table[available].round(6).to_markdown(index=False))
        f.write("\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_weights = parse_float_list(args.start_weights, name="--start-weights")
    use_norm_cost = resolve_use_norm_cost(args.target_family, args.use_norm_cost)
    use_norm_pr = parse_bool_text(args.use_norm_pr, name="--use-norm-pr")
    objective_weights = parse_objective_weights(args.objective_weights, args.target_family)

    if args.run_flexdc and args.utilization is None:
        raise ValueError("--utilization is required when --run-flexdc is used so validation matches inference.")

    run = init_wandb(args, {**vars(args), "objective_weights": objective_weights, "use_norm_cost_resolved": use_norm_cost})

    trajectory, candidate, prediction_table = optimize_inputs(
        args.model_file,
        args.workload_config,
        args.experiment_config,
        args.norm_source_results_csv,
        args.start_pbar_kw_per_server,
        args.start_r_kw_per_server,
        start_weights,
        target_family=args.target_family,
        target_mode=args.target_mode,
        raw_qos_aggregation=args.raw_qos_aggregation,
        use_norm_cost=use_norm_cost,
        use_norm_pr=bool(use_norm_pr),
        objective_weights=objective_weights,
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

    start_results, start_diagnostics = run_wizard(
        args,
        "start",
        candidate["starting_pbar_kw_per_server"],
        candidate["starting_r_kw_per_server"],
        candidate["starting_weights"],
    )
    opt_results, opt_diagnostics = run_wizard(
        args,
        "selected",
        candidate["optimized_pbar_kw_per_server"],
        candidate["optimized_r_kw_per_server"],
        candidate["optimized_weights"],
    )
    table = make_validation_table(
        objective_weights,
        prediction_table,
        start_results,
        start_diagnostics,
        opt_results,
        opt_diagnostics,
        args.target_family,
        args.target_mode,
        use_norm_cost,
        args.raw_qos_aggregation,
    )
    table.to_csv(out_dir / "end_to_end_validation_summary.csv", index=False)
    write_markdown_report(table, out_dir / "end_to_end_validation_report.md", args.target_family, args.target_mode, objective_weights)

    print("\nEnd-to-end validation summary")
    display_cols = [
        "Configuration",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "Predicted_Optimization_Objective",
        "Actual_Optimization_Objective",
        "Actual_Objective_Change_Percent_vs_Start",
        "Ctrack_Epsilon_90th",
        "QoS_Violation_Ratio",
        "Diagnostic_FullPaperObjective_Cost",
    ]
    display_cols = [col for col in display_cols if col in table.columns]
    print(table[display_cols].round(6).to_string(index=False))
    print(f"\nSaved: {out_dir / 'end_to_end_validation_summary.csv'}")
    print(f"Saved: {out_dir / 'end_to_end_validation_report.md'}")

    if run is not None:
        for _, row in table.iterrows():
            prefix = "start" if row["Configuration"].startswith("Starting") else "selected"
            log_payload = {
                f"validation/{prefix}_predicted_objective": row["Predicted_Optimization_Objective"],
                f"validation/{prefix}_actual_objective": row["Actual_Optimization_Objective"],
            }
            for col in ["Ctrack_Epsilon_90th", "QoS_Violation_Ratio", "Diagnostic_FullPaperObjective_Cost"]:
                if col in row and not pd.isna(row[col]):
                    log_payload[f"validation/{prefix}_{col}"] = row[col]
            run.log(log_payload)
        run.finish()


if __name__ == "__main__":
    main()
