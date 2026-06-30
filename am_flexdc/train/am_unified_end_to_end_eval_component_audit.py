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
    """Parse a QoS probability vector stored as a list-like CSV cell."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if pd.isna(value):
        raise ValueError("QoS_Delay_Probabilities contains NaN; cannot recompute configured objective.")
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Handles Python-list strings if any older CSV used single quotes.
        import ast
        parsed = ast.literal_eval(text)
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




def compute_condor_component_values(source: pd.Series, use_norm_cost: bool = True) -> dict:
    """Compute CONDOR-style components from raw FlexDC validation outputs.

    Raw definitions:
      CPower_raw = 0.0003 * (P_actual_watts - R_actual_watts)
      CError_raw = Mtrack_Error_MeanAbs_Watts / 1000
      CQoS_raw   = 0.8 * sum SoftPlus(60 * (q_j - 0.1))

    Released CONDOR scaling:
      CPower_scaled = 120 * CPower_raw / server_count
      CError_scaled = 200 * CError_raw / server_count
      CQoS_scaled   = CQoS_raw / workload_mix_size
    """
    from am_unified_training_utilities import (
        CONDOR_POWER_COST_COEFFICIENT,
        CONDOR_QOS_BETA,
        CONDOR_QOS_RHO,
        CONDOR_QOS_THRESHOLD,
    )

    server_count = float(source["server_count"])
    workload_mix_size = float(source["workload_mix_size"])
    p_actual_watts = float(source["P_actual_watts"])
    r_actual_watts = float(source["R_actual_watts"])
    mean_abs_watts = float(source["Mtrack_Error_MeanAbs_Watts"])

    probs = parse_probability_vector(source["QoS_Delay_Probabilities"])
    raw_power = CONDOR_POWER_COST_COEFFICIENT * (p_actual_watts - r_actual_watts)
    raw_error = mean_abs_watts / 1000.0
    raw_qos = CONDOR_QOS_BETA * float(np.sum(stable_softplus(CONDOR_QOS_RHO * (probs - CONDOR_QOS_THRESHOLD))))

    scaled_power = raw_power * 120.0 / server_count
    scaled_error = raw_error * 200.0 / server_count
    scaled_qos = raw_qos / workload_mix_size

    return {
        "Condor_Raw_CPower": raw_power,
        "Condor_Raw_CError_MeanAbsTracking_kW": raw_error,
        "Condor_Raw_CQoS": raw_qos,
        "Condor_Scaled_CPower": scaled_power,
        "Condor_Scaled_CError": scaled_error,
        "Condor_Scaled_CQoS": scaled_qos,
        "Condor_Scaled_Target_Sum": scaled_power + scaled_error + scaled_qos,
        "Condor_Raw_Target_Sum": raw_power + raw_error + raw_qos,
    }


def compute_flexdc_component_values(source: pd.Series) -> dict:
    """Return configured-objective FlexDC components from the postprocessed source row."""
    return {
        "FlexDC_M_RSR": float(source["Simulator_RSR_Total_Cost"]),
        "FlexDC_Simulator_Power_Cost": float(source["Simulator_Power_Cost"]) if "Simulator_Power_Cost" in source.index and not pd.isna(source["Simulator_Power_Cost"]) else np.nan,
        "FlexDC_Mtrack_Cost": float(source["Mtrack_Cost"]) if "Mtrack_Cost" in source.index and not pd.isna(source["Mtrack_Cost"]) else np.nan,
        "FlexDC_Ctrack_Epsilon_90th": float(source["Ctrack_Epsilon_90th"]),
        "FlexDC_Ctrack_Weighted_Cost": float(source["Ctrack_Weighted_Cost"]),
        "FlexDC_CQoS_Weighted_Cost": float(source["Diagnostic_FlexDC_SoftPlus_QoS_Cost"]),
        "FlexDC_Full_Objective": float(source["Diagnostic_FullPaperObjective_Cost"]),
    }


def add_component_audit_columns(row: dict, source: pd.Series, actual: np.ndarray, predicted: pd.Series,
                                names: list[str], objective_weights: np.ndarray,
                                target_family: str, use_norm_cost: bool) -> dict:
    """Append both formulation component values and active objective contributions."""
    condor_values = compute_condor_component_values(source, use_norm_cost=use_norm_cost)
    flexdc_values = compute_flexdc_component_values(source)
    row.update(condor_values)
    row.update(flexdc_values)

    row["Objective_Weights"] = json.dumps([float(x) for x in objective_weights])
    for idx, name in enumerate(names):
        weight = float(objective_weights[idx])
        actual_value = float(actual[idx])
        predicted_value = float(predicted[f"Predicted_{name}"])
        row[f"Active_Objective_Component_{idx}_Name"] = name
        row[f"Active_Objective_Component_{idx}_Weight"] = weight
        row[f"Active_Objective_Component_{idx}_Predicted"] = predicted_value
        row[f"Active_Objective_Component_{idx}_Actual"] = actual_value
        row[f"Active_Objective_Component_{idx}_Actual_Contribution"] = weight * actual_value
        row[f"Active_Objective_Component_{idx}_Predicted_Contribution"] = weight * predicted_value
    return row


def build_component_comparison_table(summary: pd.DataFrame, target_family: str, target_mode: str) -> pd.DataFrame:
    """Create a long-format table comparing start vs selected for all component definitions."""
    if len(summary) < 2:
        raise ValueError("Need at least start and selected rows to build component comparison table.")
    start = summary.iloc[0]
    selected = summary.iloc[1]

    active_names = {
        str(start.get(f"Active_Objective_Component_{i}_Name", "")): float(start.get(f"Active_Objective_Component_{i}_Weight", np.nan))
        for i in range(3)
    }

    rows = []

    def add_row(group, component, col, meaning="", active_name=None):
        s = float(start[col]) if col in summary.columns and not pd.isna(start[col]) else np.nan
        v = float(selected[col]) if col in summary.columns and not pd.isna(selected[col]) else np.nan
        change = v - s if not (pd.isna(s) or pd.isna(v)) else np.nan
        pct = pct_change(s, v) if not (pd.isna(s) or pd.isna(v)) else np.nan
        weight = active_names.get(active_name, np.nan) if active_name else np.nan
        rows.append({
            "Group": group,
            "Component": component,
            "Starting": s,
            "Selected": v,
            "Change": change,
            "Change_Percent": pct,
            "Active_Objective_Weight": weight,
            "Starting_Weighted_Contribution": weight * s if not pd.isna(weight) and not pd.isna(s) else np.nan,
            "Selected_Weighted_Contribution": weight * v if not pd.isna(weight) and not pd.isna(v) else np.nan,
            "Meaning": meaning,
        })

    add_row("Active optimized objective", "Predicted objective", "Predicted_Optimization_Objective", "Surrogate objective minimized during gradient descent")
    add_row("Active optimized objective", "Actual objective", "Actual_Optimization_Objective", "Same target family/mode as optimizer, recomputed from simulator output")

    add_row("CONDOR-style components", "CPower scaled", "Condor_Scaled_CPower", "Released CONDOR-scaled power component", "condor_cost_power")
    add_row("CONDOR-style components", "CError scaled", "Condor_Scaled_CError", "Released CONDOR-scaled mean tracking-deviation component", "condor_cost_error")
    add_row("CONDOR-style components", "CQoS scaled", "Condor_Scaled_CQoS", "Released CONDOR-scaled smoothed QoS component", "condor_cost_qos")
    add_row("CONDOR-style components", "CONDOR scaled target sum", "Condor_Scaled_Target_Sum", "Unweighted sum of scaled CONDOR components")

    add_row("CONDOR-style raw diagnostics", "CPower raw", "Condor_Raw_CPower", "0.0003 * (P_actual_watts - R_actual_watts)")
    add_row("CONDOR-style raw diagnostics", "CError raw mean abs tracking kW", "Condor_Raw_CError_MeanAbsTracking_kW", "Mtrack_Error_MeanAbs_Watts / 1000")
    add_row("CONDOR-style raw diagnostics", "CQoS raw", "Condor_Raw_CQoS", "0.8 * sum SoftPlus(60 * (q_j - 0.1))")

    add_row("FlexDC-style components", "M_RSR", "FlexDC_M_RSR", "Reserve-service settlement cost", "flexdc_M_RSR")
    add_row("FlexDC-style components", "Ctrack weighted", "FlexDC_Ctrack_Weighted_Cost", "Configured quantile tracking penalty", "flexdc_Ctrack_weighted")
    add_row("FlexDC-style components", "CQoS weighted", "FlexDC_CQoS_Weighted_Cost", "Configured smoothed QoS penalty", "flexdc_CQoS_weighted")
    add_row("FlexDC-style components", "Full objective", "FlexDC_Full_Objective", "M_RSR + Ctrack + CQoS")

    add_row("Raw simulator diagnostics", "p90 tracking error", "Ctrack_Epsilon_90th", "Raw p90 normalized tracking error")
    add_row("Raw simulator diagnostics", "Mean abs tracking error watts", "Mtrack_Error_MeanAbs_Watts", "Mean abs tracking error from power_trace.csv")
    add_row("Raw simulator diagnostics", "QoS violation ratio", "QoS_Violation_Ratio", "Fraction of job types violating QoS")
    add_row("Raw simulator diagnostics", "QoS delay probability sum", "QoS_Delay_Probability_Sum", "Sum of per-class delay probabilities")

    return pd.DataFrame(rows)



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

        row = add_component_audit_columns(
            row=row,
            source=source,
            actual=actual,
            predicted=predicted,
            names=names,
            objective_weights=weights,
            target_family=target_family,
            use_norm_cost=use_norm_cost,
        )

        for col in [
            "Simulator_RSR_Total_Cost",
            "Simulator_Power_Cost",
            "Mtrack_Cost",
            "Ctrack_Epsilon_90th",
            "Ctrack_Weighted_Cost",
            "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
            "Diagnostic_FullPaperObjective_Cost",
            "Mtrack_Error_MeanAbs_Normalized",
            "Mtrack_Error_MeanAbs_Watts",
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

    component_table = build_component_comparison_table(table, args.target_family, args.target_mode)
    component_table.to_csv(out_dir / "end_to_end_component_comparison.csv", index=False)
    component_table.to_markdown(out_dir / "end_to_end_component_comparison.md", index=False)

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
    print(f"Saved: {out_dir / 'end_to_end_component_comparison.csv'}")
    print(f"Saved: {out_dir / 'end_to_end_component_comparison.md'}")
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
