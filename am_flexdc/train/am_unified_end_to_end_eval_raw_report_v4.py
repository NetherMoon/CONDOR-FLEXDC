"""Run unified surrogate optimization and optional FlexDC validation (raw-report v4).

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
import re
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from am_unified_optimize_one_v2 import optimize_inputs
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
    parser.add_argument("--pbar-min-kw-per-server", type=float, default=None,
                        help="Optional explicit lower bound for Pbar in kW/server. Leave unset for broad workload-derived bounds.")
    parser.add_argument("--pbar-max-kw-per-server", type=float, default=None,
                        help="Optional explicit upper bound for Pbar in kW/server. Leave unset for broad workload-derived bounds.")
    parser.add_argument("--r-max-kw-per-server", type=float, default=None,
                        help="Optional explicit upper bound for R in kW/server. Leave unset for broad workload-derived bounds.")

    # FlexDC execution arguments.
    parser.add_argument("--flexdc-root", required=True, help="Path to FlexDC repository root.")
    parser.add_argument("--flexdc-python", default=sys.executable, help="Python executable used to run am_data_extraction_wizard.py.")
    parser.add_argument("--gradient-config", default="../../configs/gradient_descent/gradient_descent.ini")
    parser.add_argument("--cluster-config", default="../../configs/cluster/cluster.ini")
    parser.add_argument("--policy-name", default="AQA")
    parser.add_argument("--node-count-control", default="true")
    parser.add_argument("--run-flexdc", action="store_true", help="Actually run FlexDC validation. Omit for optimization-only dry run.")



    # Optional report constants for paper-form objective reconstruction.
    # For raw FlexDC models these are reporting-only; optimization still uses --objective-weights.
    parser.add_argument("--report-ctrack-psi", type=float, default=1.0)
    parser.add_argument("--report-ctrack-mu", type=float, default=10.0)
    parser.add_argument("--report-ctrack-gamma", type=float, default=0.3)
    parser.add_argument("--report-qos-beta", type=float, default=20.0)
    parser.add_argument("--report-qos-rho", type=float, default=2.0)
    parser.add_argument("--report-qos-threshold", type=float, default=0.1)

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


def run_wizard(args: argparse.Namespace, label: str, pbar: float, reserve: float, weights: list[float]) -> tuple[Path, Path, Path]:
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
    return folder / "grid_search_results.csv", folder / "grid_search_diagnostics.csv", folder



# Configured FlexDC objective constants from the reference simulated-annealing cost configuration.
# These are applied to the one-row FlexDC validation outputs after the simulator runs so that
# end-to-end evaluation uses the same objective labels as the recomputed training dataset.
FLEXDC_CTRACK_PSI = 1.0
FLEXDC_CTRACK_MU = 10.0
FLEXDC_CTRACK_GAMMA = 0.3
FLEXDC_QOS_BETA = 20.0
FLEXDC_QOS_RHO = 2.0
FLEXDC_QOS_THRESHOLD = 0.1


def stable_softplus(x):
    """Numerically stable SoftPlus for scalars or numpy arrays."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def parse_probability_vector(value) -> np.ndarray:
    """Parse a QoS probability vector stored as a list-like CSV cell.

    Handles JSON lists, Python-list strings, and older strings like
    [np.float64(0.1), ...].
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if pd.isna(value):
        raise ValueError("QoS_Delay_Probabilities contains NaN; cannot parse probability vector.")
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        import ast
        cleaned = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
        parsed = ast.literal_eval(cleaned)
    arr = np.asarray(parsed, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D QoS probability vector, got shape {arr.shape}")
    return arr


def apply_configured_flexdc_objective(df: pd.DataFrame) -> pd.DataFrame:
    """Replace FlexDC penalty columns with configured objective constants.

    This keeps downstream training/inference code unchanged because the usual
    column names are overwritten with the configured objective values:
      - Ctrack_Weighted_Cost
      - Diagnostic_FlexDC_SoftPlus_QoS_Cost
      - Diagnostic_FullPaperObjective_Cost
    """
    required = ["Simulator_RSR_Total_Cost", "Ctrack_Epsilon_90th", "QoS_Delay_Probabilities"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Cannot apply configured FlexDC objective; missing columns: {missing}")

    out = df.copy()
    eps = out["Ctrack_Epsilon_90th"].astype(float).to_numpy()
    ctrack_residual = eps - FLEXDC_CTRACK_GAMMA
    ctrack_scaled = FLEXDC_CTRACK_MU * ctrack_residual
    ctrack_softplus = stable_softplus(ctrack_scaled)
    ctrack_weighted = FLEXDC_CTRACK_PSI * ctrack_softplus

    probs = [parse_probability_vector(v) for v in out["QoS_Delay_Probabilities"]]
    residuals = [p - FLEXDC_QOS_THRESHOLD for p in probs]
    qos_softplus_sum = np.asarray([
        float(np.sum(stable_softplus(FLEXDC_QOS_RHO * r))) for r in residuals
    ], dtype=float)
    qos_weighted = FLEXDC_QOS_BETA * qos_softplus_sum

    out["Ctrack_Gamma"] = FLEXDC_CTRACK_GAMMA
    out["Ctrack_Psi"] = FLEXDC_CTRACK_PSI
    out["Ctrack_Mu"] = FLEXDC_CTRACK_MU
    out["Ctrack_Residual"] = ctrack_residual
    out["Ctrack_MuScaled_Residual"] = ctrack_scaled
    out["Ctrack_SoftPlus_Value"] = ctrack_softplus
    out["Ctrack_Weighted_Cost"] = ctrack_weighted
    out["QoS_Delay_Probability_Residuals"] = [json.dumps([float(x) for x in r]) for r in residuals]
    out["QoS_Delay_Probability_Residual_Sum"] = [float(np.sum(r)) for r in residuals]
    out["Diagnostic_FlexDC_SoftPlus_QoS_Cost"] = qos_weighted
    out["Diagnostic_FullPaperObjective_Cost"] = (
        out["Simulator_RSR_Total_Cost"].astype(float).to_numpy()
        + ctrack_weighted
        + qos_weighted
    )
    return out

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

    # The FlexDC wizard writes objective columns using the gradient config it was run with.
    # Recompute/replace those objective columns here so validation matches the configured
    # training dataset without changing the downstream target-building code.
    df = apply_configured_flexdc_objective(df)

    targets, names = build_targets(df, target_family, target_mode, use_norm_cost, raw_qos_aggregation)
    return df.iloc[0], targets[0].astype(float), names


def pct_change(start: float, selected: float) -> float:
    if abs(start) < 1e-12:
        return float("nan")
    return float((selected - start) / abs(start) * 100.0)




def paper_form_objective_from_raw(
    m_rsr: float,
    epsilon_90: float,
    qos_probs: np.ndarray,
    *,
    ctrack_psi: float,
    ctrack_mu: float,
    ctrack_gamma: float,
    qos_beta: float,
    qos_rho: float,
    qos_threshold: float,
) -> dict:
    """Compute paper-form objective components from raw metrics."""
    qos_probs = np.asarray(qos_probs, dtype=float)
    ctrack = float(ctrack_psi * stable_softplus(ctrack_mu * (float(epsilon_90) - ctrack_gamma)))
    cqos = float(qos_beta * np.sum(stable_softplus(qos_rho * (qos_probs - qos_threshold))))
    return {
        "paper_M_RSR": float(m_rsr),
        "paper_Ctrack_from_raw": ctrack,
        "paper_CQoS_from_raw": cqos,
        "paper_objective_from_raw": float(m_rsr) + ctrack + cqos,
    }


def approx_paper_objective_from_raw_prediction(
    pred_targets: dict,
    names: list[str],
    workload_size: int,
    *,
    ctrack_psi: float,
    ctrack_mu: float,
    ctrack_gamma: float,
    qos_beta: float,
    qos_rho: float,
    qos_threshold: float,
) -> dict:
    """Approximate paper objective from raw-model predictions.

    With current raw FlexDC model the QoS prediction is an aggregate
    mean/sum, not a per-job-type vector. For target raw_qos_probability_mean,
    this approximates every job type as having the predicted mean probability.
    """
    out = {
        "Predicted_PaperObjective_Approx": np.nan,
        "Predicted_Ctrack_Approx": np.nan,
        "Predicted_CQoS_Approx": np.nan,
    }
    if "flexdc_M_RSR" not in names or "raw_Ctrack_Epsilon_90th" not in names:
        return out
    m = float(pred_targets.get("flexdc_M_RSR", np.nan))
    eps = float(pred_targets.get("raw_Ctrack_Epsilon_90th", np.nan))
    qos_key_mean = "raw_qos_probability_mean"
    qos_key_sum = "raw_qos_probability_sum"
    if qos_key_mean in pred_targets:
        q_mean = float(pred_targets[qos_key_mean])
        q_probs = np.full(int(workload_size), q_mean, dtype=float)
    elif qos_key_sum in pred_targets:
        q_sum = float(pred_targets[qos_key_sum])
        q_probs = np.full(int(workload_size), q_sum / float(workload_size), dtype=float)
    else:
        return out
    pieces = paper_form_objective_from_raw(
        m, eps, q_probs,
        ctrack_psi=ctrack_psi, ctrack_mu=ctrack_mu, ctrack_gamma=ctrack_gamma,
        qos_beta=qos_beta, qos_rho=qos_rho, qos_threshold=qos_threshold,
    )
    out["Predicted_PaperObjective_Approx"] = pieces["paper_objective_from_raw"]
    out["Predicted_Ctrack_Approx"] = pieces["paper_Ctrack_from_raw"]
    out["Predicted_CQoS_Approx"] = pieces["paper_CQoS_from_raw"]
    return out

def make_validation_table(
    objective_weights: list[float],
    prediction_table: pd.DataFrame,
    start_results: Path,
    start_diagnostics: Path,
    start_folder: Path,
    opt_results: Path,
    opt_diagnostics: Path,
    opt_folder: Path,
    target_family: str,
    target_mode: str,
    use_norm_cost: bool,
    raw_qos_aggregation: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    weights = np.asarray(objective_weights, dtype=float)
    pred = prediction_table.set_index("Configuration")
    names = target_names(target_family, target_mode, raw_qos_aggregation)
    rows = []
    for label, result_path, diag_path, folder_path, pred_label in [
        ("Starting configuration", start_results, start_diagnostics, start_folder, "Starting configuration"),
        ("Selected configuration", opt_results, opt_diagnostics, opt_folder, "Selected configuration"),
    ]:
        source, actual, actual_names = actual_targets_from_flexdc(
            result_path, diag_path, target_family, target_mode, use_norm_cost, raw_qos_aggregation
        )
        if actual_names != names:
            raise ValueError(f"Target-name mismatch: predicted={names}, actual={actual_names}")
        predicted = pred.loc[pred_label]
        # Keep only actual workload-weight columns from grid_search_results.csv.
        # The merged diagnostics row also contains audit metadata such as
        # Weight_Equal_Value and Weight_Final_Lower_Bound; those are not
        # workload weights and should not be parsed as Weight_i.
        weight_cols = sorted(
            [
                col for col in source.index
                if str(col).startswith("Weight_") and str(col).split("_")[-1].isdigit()
            ],
            key=lambda name: int(str(name).split("_")[-1]),
        )
        qos_probs = parse_probability_vector(source["QoS_Delay_Probabilities"]) if "QoS_Delay_Probabilities" in source.index else np.asarray([], dtype=float)
        pred_targets = {name: float(predicted[f"Predicted_{name}"]) for name in names if f"Predicted_{name}" in predicted.index}
        pred_report = approx_paper_objective_from_raw_prediction(
            pred_targets, names, int(source.get("workload_mix_size", len(qos_probs) if len(qos_probs) else 0)),
            ctrack_psi=args.report_ctrack_psi,
            ctrack_mu=args.report_ctrack_mu,
            ctrack_gamma=args.report_ctrack_gamma,
            qos_beta=args.report_qos_beta,
            qos_rho=args.report_qos_rho,
            qos_threshold=args.report_qos_threshold,
        )
        actual_report = paper_form_objective_from_raw(
            float(source["Simulator_RSR_Total_Cost"]),
            float(source["Ctrack_Epsilon_90th"]),
            qos_probs,
            ctrack_psi=args.report_ctrack_psi,
            ctrack_mu=args.report_ctrack_mu,
            ctrack_gamma=args.report_ctrack_gamma,
            qos_beta=args.report_qos_beta,
            qos_rho=args.report_qos_rho,
            qos_threshold=args.report_qos_threshold,
        ) if len(qos_probs) else {}

        row = {
            "Configuration": label,
            "FlexDC_Output_Dir": str(folder_path),
            "FlexDC_Results_CSV": str(result_path),
            "FlexDC_Diagnostics_CSV": str(diag_path),
            "Pbar_kw_per_server": float(source["Pbar_kw_per_server"]),
            "R_kw_per_server": float(source["R_kw_per_server"]),
            "Pbar_plus_R": float(source["Pbar_kw_per_server"]) + float(source["R_kw_per_server"]),
            "Pbar_minus_R": float(source["Pbar_kw_per_server"]) - float(source["R_kw_per_server"]),
            "Weights": json.dumps([float(source[col]) for col in weight_cols]),
            "Predicted_Optimization_Objective": float(predicted["Predicted_Optimization_Objective"]),
            "Actual_Optimization_Objective": float(np.dot(weights, actual)),
            "Predicted_Target_Sum": float(predicted["Predicted_Target_Sum"]),
            "Actual_Target_Sum": float(np.sum(actual)),
            "QoS_Delay_Probabilities": json.dumps([float(x) for x in qos_probs]) if len(qos_probs) else "",
            "Max_QoS_Delay_Probability": float(np.max(qos_probs)) if len(qos_probs) else np.nan,
            "Mean_QoS_Delay_Probability": float(np.mean(qos_probs)) if len(qos_probs) else np.nan,
            "Tracking_Pass": bool(float(source["Ctrack_Epsilon_90th"]) <= args.report_ctrack_gamma) if "Ctrack_Epsilon_90th" in source.index else False,
            "QoS_Pass_CurrentLogic": bool(float(source["QoS_Violation_Ratio"]) <= args.report_qos_threshold) if "QoS_Violation_Ratio" in source.index else False,
            **pred_report,
            **actual_report,
        }
        row["Both_Pass_CurrentLogic"] = bool(row["Tracking_Pass"] and row["QoS_Pass_CurrentLogic"])
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
        pbar_min_kw_per_server=args.pbar_min_kw_per_server,
        pbar_max_kw_per_server=args.pbar_max_kw_per_server,
        r_max_kw_per_server=args.r_max_kw_per_server,
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

    start_results, start_diagnostics, start_folder = run_wizard(
        args,
        "start",
        candidate["starting_pbar_kw_per_server"],
        candidate["starting_r_kw_per_server"],
        candidate["starting_weights"],
    )
    opt_results, opt_diagnostics, opt_folder = run_wizard(
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
        start_folder,
        opt_results,
        opt_diagnostics,
        opt_folder,
        args.target_family,
        args.target_mode,
        use_norm_cost,
        args.raw_qos_aggregation,
        args,
    )
    table.to_csv(out_dir / "end_to_end_validation_summary.csv", index=False)
    key_cols = [c for c in [
        "Configuration", "Pbar_kw_per_server", "R_kw_per_server", "Pbar_plus_R", "Pbar_minus_R", "Weights",
        "Predicted_flexdc_M_RSR", "Actual_flexdc_M_RSR",
        "Predicted_raw_Ctrack_Epsilon_90th", "Actual_raw_Ctrack_Epsilon_90th",
        "Predicted_raw_qos_probability_mean", "Actual_raw_qos_probability_mean",
        "QoS_Violation_Ratio", "Max_QoS_Delay_Probability", "Mean_QoS_Delay_Probability", "Tracking_Pass", "QoS_Pass_CurrentLogic", "Both_Pass_CurrentLogic",
        "Predicted_PaperObjective_Approx", "paper_objective_from_raw",
        "FlexDC_Output_Dir",
    ] if c in table.columns]
    table[key_cols].to_csv(out_dir / "end_to_end_raw_key_summary.csv", index=False)
    write_markdown_report(table, out_dir / "end_to_end_validation_report.md", args.target_family, args.target_mode, objective_weights)

    print("\nEnd-to-end validation summary")
    display_cols = [
        "Configuration",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "Predicted_Optimization_Objective",
        "Actual_Optimization_Objective",
        "Predicted_flexdc_M_RSR",
        "Actual_flexdc_M_RSR",
        "Predicted_raw_Ctrack_Epsilon_90th",
        "Actual_raw_Ctrack_Epsilon_90th",
        "Predicted_raw_qos_probability_mean",
        "Actual_raw_qos_probability_mean",
        "QoS_Violation_Ratio",
        "Max_QoS_Delay_Probability",
        "Mean_QoS_Delay_Probability",
        "Both_Pass_CurrentLogic",
        "paper_objective_from_raw",
    ]
    display_cols = [col for col in display_cols if col in table.columns]
    print(table[display_cols].round(6).to_string(index=False))
    print(f"\nSaved: {out_dir / 'end_to_end_validation_summary.csv'}")
    print(f"Saved: {out_dir / 'end_to_end_validation_report.md'}")
    print(f"Saved: {out_dir / 'end_to_end_raw_key_summary.csv'}")

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
