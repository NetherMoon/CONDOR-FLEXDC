"""Orchestrate CONDOR-style optimization and FlexDC validation.

This script runs the frozen CONDOR/FlexDC model to select P,R,w, then runs the
FlexDC data extraction wizard for the starting and selected configurations, and
finally compares predicted CONDOR-style costs with actual FlexDC-derived costs.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import textwrap
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


START_LABEL = "Starting configuration"
SELECTED_LABEL = "CONDOR-selected configuration"


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


def _read_one_row(path: Path, label: str) -> pd.Series:
    data = pd.read_csv(path)
    if len(data) != 1:
        raise ValueError(f"Expected one {label} row in {path}, found {len(data)}")
    return data.iloc[0]


def labels_from_results_csv(path: Path) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    row = _read_one_row(path, "FlexDC result")
    n = float(row["server_count"])
    j = float(row["workload_mix_size"])
    raw_power = CONDOR_POWER_COST_COEFFICIENT * (float(row["P_actual_watts"]) - float(row["R_actual_watts"]))
    raw_error = float(row["Mtrack_Error_MeanAbs_Watts"]) / 1000.0
    probabilities = np.asarray(json.loads(row["QoS_Delay_Probabilities"]), dtype=float)
    raw_qos = CONDOR_QOS_BETA * np.logaddexp(0, CONDOR_QOS_RHO * (probabilities - CONDOR_QOS_THRESHOLD)).sum()
    scaled = np.asarray([raw_power * 120.0 / n, raw_error * 200.0 / n, raw_qos / j])
    return row, np.asarray([raw_power, raw_error, raw_qos]), scaled


def _optional_float(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def make_validation_table(
    cost_weights: list[float],
    prediction_table: pd.DataFrame,
    start_results: Path,
    opt_results: Path,
    start_diagnostics: Path | None = None,
    opt_diagnostics: Path | None = None,
) -> pd.DataFrame:
    weights = np.asarray(cost_weights, dtype=float)
    pred = prediction_table.set_index("Configuration")
    rows = []
    run_specs = [
        (START_LABEL, start_results, start_diagnostics),
        (SELECTED_LABEL, opt_results, opt_diagnostics),
    ]
    for label, results_path, diagnostics_path in run_specs:
        source, raw, scaled = labels_from_results_csv(results_path)
        diagnostics = _read_one_row(diagnostics_path, "FlexDC diagnostics") if diagnostics_path is not None else None
        predicted = pred.loc[label]
        weight_cols = sorted(
            [col for col in source.index if col.startswith("Weight_") and col != "Weight_Sample_ID"],
            key=lambda name: int(name.split("_")[-1]),
        )
        flexdc_rsr = float(source["Simulator_RSR_Total_Cost"])
        flexdc_power = float(source["Simulator_Power_Cost"])
        flexdc_mtrack = float(source["Mtrack_Cost"])
        flexdc_ctrack = _optional_float(diagnostics, "Ctrack_Weighted_Cost")
        flexdc_cqos = _optional_float(diagnostics, "Diagnostic_FlexDC_SoftPlus_QoS_Cost")
        flexdc_full = _optional_float(diagnostics, "Diagnostic_FullPaperObjective_Cost")
        if flexdc_full is None and flexdc_ctrack is not None and flexdc_cqos is not None:
            flexdc_full = flexdc_rsr + flexdc_ctrack + flexdc_cqos
        rows.append({
            "Configuration": label,
            "Pbar_kw_per_server": float(source["Pbar_kw_per_server"]),
            "R_kw_per_server": float(source["R_kw_per_server"]),
            "P_minus_R_kw_per_server": float(source["Pbar_kw_per_server"]) - float(source["R_kw_per_server"]),
            "P_plus_R_kw_per_server": float(source["Pbar_kw_per_server"]) + float(source["R_kw_per_server"]),
            "Weights": json.dumps([float(source[col]) for col in weight_cols]),
            "Predicted_Optimization_Objective": float(predicted["Predicted_Optimization_Objective"]),
            "Actual_Optimization_Objective": float(np.dot(weights, scaled)),
            "Optimization_Objective_Prediction_Error": float(np.dot(weights, scaled)) - float(predicted["Predicted_Optimization_Objective"]),
            "Predicted_Scaled_cost_power": float(predicted["Predicted_Scaled_cost_power"]),
            "Actual_Scaled_cost_power": scaled[0],
            "Predicted_Scaled_cost_error": float(predicted["Predicted_Scaled_cost_error"]),
            "Actual_Scaled_cost_error": scaled[1],
            "Predicted_Scaled_cost_qos": float(predicted["Predicted_Scaled_cost_qos"]),
            "Actual_Scaled_cost_qos": scaled[2],
            "Predicted_Raw_cost_power": float(predicted["Predicted_Raw_cost_power"]),
            "Actual_Raw_cost_power": raw[0],
            "Predicted_Raw_cost_error": float(predicted["Predicted_Raw_cost_error"]),
            "Actual_Raw_cost_error": raw[1],
            "Predicted_Raw_cost_qos": float(predicted["Predicted_Raw_cost_qos"]),
            "Actual_Raw_cost_qos": raw[2],
            "Predicted_Raw_cost_sum": float(predicted["Predicted_Raw_cost_power"] + predicted["Predicted_Raw_cost_error"] + predicted["Predicted_Raw_cost_qos"]),
            "Actual_Raw_cost_sum": float(raw.sum()),
            "FlexDC_RSR_Total_Cost": flexdc_rsr,
            "FlexDC_Simulator_Power_Cost": flexdc_power,
            "FlexDC_Mtrack_Cost": flexdc_mtrack,
            "FlexDC_Ctrack_Weighted_Cost": flexdc_ctrack,
            "FlexDC_CQoS_SoftPlus_Cost": flexdc_cqos,
            "FlexDC_Full_Objective_Cost": flexdc_full,
            "MeanAbs_Normalized_Tracking_Error": float(source["Mtrack_Error_MeanAbs_Normalized"]),
            "Ctrack_Epsilon_90th": _optional_float(diagnostics, "Ctrack_Epsilon_90th"),
            "QoS_Violation_Ratio": float(source["QoS_Violation_Ratio"]),
            "QoS_Delay_Probability_Sum": float(source["QoS_Delay_Probability_Sum"]),
            "FlexDC_Results_CSV": str(results_path),
            "FlexDC_Diagnostics_CSV": str(diagnostics_path) if diagnostics_path is not None else "",
        })
    return pd.DataFrame(rows)


def _pct_change(start_value, selected_value):
    if not isinstance(start_value, (int, float, np.floating)) or not isinstance(selected_value, (int, float, np.floating)):
        return np.nan
    if pd.isna(start_value) or pd.isna(selected_value):
        return np.nan
    if abs(float(start_value)) < 1e-12:
        return 0.0 if abs(float(selected_value)) < 1e-12 else np.nan
    return (float(selected_value) - float(start_value)) / abs(float(start_value)) * 100.0


def _round_for_report(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float, np.floating)):
        return round(float(value), 6)
    return value


def _metric_status(start_value, selected_value, direction: str) -> str:
    if direction == "context":
        return "context"
    if not isinstance(start_value, (int, float, np.floating)) or not isinstance(selected_value, (int, float, np.floating)):
        return ""
    if pd.isna(start_value) or pd.isna(selected_value):
        return ""
    delta = float(selected_value) - float(start_value)
    if abs(delta) < 1e-12:
        return "same"
    if direction == "lower":
        return "improved" if delta < 0 else "worse"
    if direction == "higher":
        return "improved" if delta > 0 else "worse"
    if direction == "abs_lower":
        return "improved" if abs(float(selected_value)) < abs(float(start_value)) else "worse"
    return ""


def make_report_tables(table: pd.DataFrame, cost_weights: list[float], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = table.set_index("Configuration").loc[START_LABEL]
    selected = table.set_index("Configuration").loc[SELECTED_LABEL]

    overview_columns = [
        "Configuration",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "P_minus_R_kw_per_server",
        "Weights",
        "Predicted_Optimization_Objective",
        "Actual_Optimization_Objective",
        "FlexDC_Full_Objective_Cost",
        "Ctrack_Epsilon_90th",
        "QoS_Violation_Ratio",
        "MeanAbs_Normalized_Tracking_Error",
    ]
    overview = table[overview_columns].copy()

    # Direction controls color/status in the report table.
    # "lower" means a decrease is good. "context" means no good/bad interpretation.
    metric_specs = [
        ("Configuration", "Pbar kW/server", "Pbar_kw_per_server", "context"),
        ("Configuration", "R kW/server", "R_kw_per_server", "context"),
        ("Configuration", "Pbar - R kW/server", "P_minus_R_kw_per_server", "context"),
        ("Configuration", "Pbar + R kW/server", "P_plus_R_kw_per_server", "context"),
        ("CONDOR objective", f"Predicted objective ({cost_weights})", "Predicted_Optimization_Objective", "lower"),
        ("CONDOR objective", f"Actual objective ({cost_weights})", "Actual_Optimization_Objective", "lower"),
        ("CONDOR objective", "Prediction error: actual - predicted", "Optimization_Objective_Prediction_Error", "abs_lower"),
        ("CONDOR scaled components", "Predicted scaled cost_power", "Predicted_Scaled_cost_power", "lower"),
        ("CONDOR scaled components", "Actual scaled cost_power", "Actual_Scaled_cost_power", "lower"),
        ("CONDOR scaled components", "Predicted scaled cost_error", "Predicted_Scaled_cost_error", "lower"),
        ("CONDOR scaled components", "Actual scaled cost_error", "Actual_Scaled_cost_error", "lower"),
        ("CONDOR scaled components", "Predicted scaled cost_qos", "Predicted_Scaled_cost_qos", "lower"),
        ("CONDOR scaled components", "Actual scaled cost_qos", "Actual_Scaled_cost_qos", "lower"),
        ("CONDOR unscaled components", "Predicted raw cost_power", "Predicted_Raw_cost_power", "lower"),
        ("CONDOR unscaled components", "Actual raw cost_power", "Actual_Raw_cost_power", "lower"),
        ("CONDOR unscaled components", "Predicted raw cost_error", "Predicted_Raw_cost_error", "lower"),
        ("CONDOR unscaled components", "Actual raw cost_error", "Actual_Raw_cost_error", "lower"),
        ("CONDOR unscaled components", "Predicted raw cost_qos", "Predicted_Raw_cost_qos", "lower"),
        ("CONDOR unscaled components", "Actual raw cost_qos", "Actual_Raw_cost_qos", "lower"),
        ("CONDOR unscaled components", "Predicted raw component sum", "Predicted_Raw_cost_sum", "lower"),
        ("CONDOR unscaled components", "Actual raw component sum", "Actual_Raw_cost_sum", "lower"),
        ("FlexDC objective", "FlexDC full objective", "FlexDC_Full_Objective_Cost", "lower"),
        ("FlexDC objective", "FlexDC RSR total cost", "FlexDC_RSR_Total_Cost", "lower"),
        ("FlexDC objective", "FlexDC simulator power cost", "FlexDC_Simulator_Power_Cost", "lower"),
        ("FlexDC objective", "FlexDC monetary tracking cost", "FlexDC_Mtrack_Cost", "lower"),
        ("FlexDC objective", "FlexDC Ctrack weighted cost", "FlexDC_Ctrack_Weighted_Cost", "lower"),
        ("FlexDC objective", "FlexDC CQoS SoftPlus cost", "FlexDC_CQoS_SoftPlus_Cost", "lower"),
        ("Diagnostics", "Mean abs normalized tracking error", "MeanAbs_Normalized_Tracking_Error", "lower"),
        ("Diagnostics", "90th percentile tracking error", "Ctrack_Epsilon_90th", "lower"),
        ("Diagnostics", "QoS violation ratio", "QoS_Violation_Ratio", "lower"),
        ("Diagnostics", "QoS delay probability sum", "QoS_Delay_Probability_Sum", "lower"),
    ]
    rows = []
    for category, metric, column, direction in metric_specs:
        start_value = start[column]
        selected_value = selected[column]
        delta = selected_value - start_value if isinstance(start_value, (int, float, np.floating)) else np.nan
        pct = _pct_change(start_value, selected_value)
        status = _metric_status(start_value, selected_value, direction)
        rows.append({
            "Category": category,
            "Metric": metric,
            "Starting": _round_for_report(start_value),
            "CONDOR-selected": _round_for_report(selected_value),
            "Change": _round_for_report(delta),
            "Percent_Change": "" if pd.isna(pct) else f"{pct:.2f}%",
            "Status": status,
        })
    report = pd.DataFrame(rows)
    return overview, report

def _markdown_from_df(df: pd.DataFrame) -> str:
    display_df = df.fillna("").astype(str)
    headers = list(display_df.columns)
    rows = [headers] + display_df.values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    header = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows[1:]]
    return "\n".join([header, sep] + body)


def write_report_markdown(path: Path, overview: pd.DataFrame, report: pd.DataFrame, args: argparse.Namespace, cost_weights: list[float]) -> None:
    config_text = (
        f"workload={Path(args.workload_config).stem}, "
        f"server_count={args.server_count}, utilization={args.utilization}, "
        f"cost_weights={cost_weights}, iterations={args.iterations}, lr={args.lr}"
    )
    actual_obj = report[(report["Category"] == "CONDOR objective") & (report["Metric"].str.startswith("Actual objective"))]
    flexdc_obj = report[(report["Category"] == "FlexDC objective") & (report["Metric"] == "FlexDC full objective")]
    tracking_p90 = report[(report["Category"] == "Diagnostics") & (report["Metric"] == "90th percentile tracking error")]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# End-to-End CONDOR/FlexDC Validation Report\n\n")
        f.write(f"Config: `{config_text}`\n\n")
        f.write("## Quick read\n\n")
        if not actual_obj.empty:
            row = actual_obj.iloc[0]
            f.write(f"- **Actual CONDOR objective:** {row['Starting']} → {row['CONDOR-selected']} ({row['Percent_Change']}, {row['Status']}).\n")
        if not flexdc_obj.empty:
            row = flexdc_obj.iloc[0]
            f.write(f"- **FlexDC full objective:** {row['Starting']} → {row['CONDOR-selected']} ({row['Percent_Change']}, {row['Status']}).\n")
        if not tracking_p90.empty:
            row = tracking_p90.iloc[0]
            f.write(f"- **p90 tracking error:** {row['Starting']} → {row['CONDOR-selected']} ({row['Percent_Change']}, {row['Status']}).\n")
        f.write("\n## Overview\n\n")
        f.write(_markdown_from_df(overview.round(6)))
        f.write("\n\n## Detailed metric changes\n\n")
        f.write(_markdown_from_df(report))
        f.write("\n")

def _safe_tag(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def _format_weight_list(value, precision: int = 6) -> str:
    """Format a JSON/list-like weight vector so it fits in PNG table cells."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    parsed = None
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    if parsed is None:
        return str(value)
    try:
        vals = [float(x) for x in parsed]
    except Exception:
        return str(value)
    # Two weights per line keeps the table compact while avoiding clipped arrays.
    pieces = [f"w{i}={v:.{precision}f}" for i, v in enumerate(vals)]
    lines = []
    for i in range(0, len(pieces), 2):
        lines.append(", ".join(pieces[i:i+2]))
    return "\n".join(lines)


def _display_cell_value(value, max_len: int = 42, column_name: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float, np.floating)):
        return f"{float(value):.6g}"

    if "weight" in str(column_name).lower():
        return _format_weight_list(value)

    text = str(value)
    # Preserve manually formatted multi-line values.
    if "\n" in text:
        wrapped_lines = []
        for line in text.splitlines():
            wrapped_lines.extend(textwrap.wrap(line, width=max_len, break_long_words=False) or [""])
        return "\n".join(wrapped_lines)

    # Do not ellipsize metric names or arrays; wrap them so nothing is hidden.
    return "\n".join(textwrap.wrap(text, width=max_len, break_long_words=False))


def _status_color(status: str) -> str:
    return {
        "improved": "#DCFCE7",
        "worse": "#FEE2E2",
        "same": "#E5E7EB",
        "context": "#F8FAFC",
    }.get(str(status).lower(), "#FFFFFF")


def _category_color(category: str) -> str:
    return {
        "Configuration": "#F3F4F6",
        "CONDOR objective": "#DBEAFE",
        "CONDOR scaled components": "#E0F2FE",
        "CONDOR unscaled components": "#ECFDF5",
        "FlexDC objective": "#FEF3C7",
        "Diagnostics": "#FCE7F3",
    }.get(str(category), "#FFFFFF")


def _rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Category": "Section",
        "Pbar_kw_per_server": "Pbar\n(kW/server)",
        "R_kw_per_server": "R\n(kW/server)",
        "P_minus_R_kw_per_server": "Pbar - R",
        "P_plus_R_kw_per_server": "Pbar + R",
        "Predicted_Optimization_Objective": "Predicted\nobjective",
        "Actual_Optimization_Objective": "Actual\nobjective",
        "Optimization_Objective_Prediction_Error": "Prediction\nerror",
        "FlexDC_Full_Objective_Cost": "FlexDC full\nobjective",
        "Ctrack_Epsilon_90th": "p90 tracking\nerror",
        "QoS_Violation_Ratio": "QoS violation\nratio",
        "MeanAbs_Normalized_Tracking_Error": "Mean tracking\nerror",
        "Percent_Change": "% change",
        "CONDOR-selected": "Selected",
    }
    return df.rename(columns=rename)


def _column_wrap_width(column_name: str, default_width: int) -> int:
    name = str(column_name).lower()
    if "section" in name:
        return 16
    if "metric" in name:
        return 34
    if "configuration" in name:
        return 20
    if "weight" in name:
        return 28
    if "objective" in name:
        return 16
    return default_width


def save_table_png(
    df: pd.DataFrame,
    path: Path,
    title: str,
    max_col_width: int = 42,
    style: str = "generic",
) -> None:
    """Save a dataframe as a readable PNG table with light styling.

    The report-style PNG intentionally separates the Section column from the rest
    of the table and blanks repeated section labels. This keeps categories useful
    without making every row visually noisy.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Warning: matplotlib is not installed; skipping PNG table {path}")
        return

    display_df = _rename_for_display(df.copy()).fillna("")

    # For report PNGs, make the category/section less distracting: show it only
    # once per group and insert a slim spacer column after it.
    spacer_col_name = " "
    if style == "report" and "Section" in display_df.columns:
        section_values = list(display_df["Section"])
        compact_sections = []
        previous = None
        for value in section_values:
            if value == previous:
                compact_sections.append("")
            else:
                compact_sections.append(value)
                previous = value
        display_df["Section"] = compact_sections
        cols = list(display_df.columns)
        insert_at = cols.index("Section") + 1
        display_df.insert(insert_at, spacer_col_name, "")

    # Format values after display-only column changes.
    for col in display_df.columns:
        width = _column_wrap_width(col, max_col_width)
        display_df[col] = display_df[col].map(lambda value, col=col, width=width: _display_cell_value(value, max_len=width, column_name=col))

    rows = max(len(display_df), 1)
    cols = max(len(display_df.columns), 1)

    # Estimate row height from wrapped line count so long metric names and weights do not clip.
    max_lines_per_row = []
    for _, row in display_df.iterrows():
        max_lines_per_row.append(max(str(x).count("\n") + 1 for x in row.values))
    line_bonus = sum(max(0, line_count - 1) for line_count in max_lines_per_row) * 0.12

    fig_width = min(max(12.5, cols * 1.75), 40)
    fig_height = min(max(3.8, rows * 0.52 + line_bonus + 2.4), 40)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=18, loc="left")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6 if cols >= 7 else 8.4)
    table.scale(1, 1.65)

    # Header styling.
    for col_idx in range(cols):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#111827")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#374151")

    col_names = list(display_df.columns)
    status_col = col_names.index("Status") if "Status" in col_names else None
    section_col = col_names.index("Section") if "Section" in col_names else None
    spacer_col = col_names.index(spacer_col_name) if spacer_col_name in col_names else None
    metric_col = col_names.index("Metric") if "Metric" in col_names else None
    config_col = col_names.index("Configuration") if "Configuration" in col_names else None

    # Manual column widths make long arrays and metric names much less cramped.
    width_map = {}
    if style == "report" and section_col is not None:
        width_map[section_col] = 0.14
    if spacer_col is not None:
        width_map[spacer_col] = 0.025
    if metric_col is not None:
        width_map[metric_col] = 0.29
    if config_col is not None:
        width_map[config_col] = 0.18
    for idx, name in enumerate(col_names):
        lower = str(name).lower()
        if "weight" in lower:
            width_map[idx] = 0.28
        elif "objective" in lower:
            width_map[idx] = 0.12
        elif idx not in width_map:
            width_map[idx] = 0.11 if cols >= 7 else 0.14
    for (row_idx, col_idx), cell in table.get_celld().items():
        if col_idx in width_map:
            cell.set_width(width_map[col_idx])

    key_metric_names = {
        "Actual objective ([0.05, 0.7, 2.0])",
        "FlexDC full objective",
        "90th percentile tracking error",
        "QoS violation ratio",
    }

    for row_idx in range(1, rows + 1):
        base_color = "#FFFFFF" if row_idx % 2 else "#F9FAFB"
        original_row = df.iloc[row_idx - 1]
        if style == "overview" and config_col is not None:
            config_text = str(original_row.get("Configuration", ""))
            base_color = "#EFF6FF" if config_text.startswith("Starting") else "#ECFDF5"
        if style == "report":
            base_color = _category_color(original_row.get("Category", ""))

        for col_idx in range(cols):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(base_color)
            cell.set_edgecolor("#D1D5DB")

        # Category/section column is useful but should not dominate the table.
        if style == "report" and section_col is not None:
            section_text = str(display_df.iloc[row_idx - 1].get("Section", ""))
            cell = table[(row_idx, section_col)]
            cell.set_facecolor("#F8FAFC" if section_text else base_color)
            cell.set_text_props(weight="bold" if section_text else "normal", color="#374151")
            if spacer_col is not None:
                spacer = table[(row_idx, spacer_col)]
                spacer.set_facecolor("#FFFFFF")
                spacer.set_edgecolor("#FFFFFF")
                spacer.get_text().set_text("")
        if spacer_col is not None:
            table[(0, spacer_col)].set_facecolor("#FFFFFF")
            table[(0, spacer_col)].set_edgecolor("#FFFFFF")
            table[(0, spacer_col)].get_text().set_text("")

        if style == "report" and status_col is not None:
            status = str(original_row.get("Status", ""))
            table[(row_idx, status_col)].set_facecolor(_status_color(status))
            table[(row_idx, status_col)].set_text_props(weight="bold")
            if status in ("improved", "worse"):
                for name in ("Selected", "Change", "% change"):
                    if name in col_names:
                        table[(row_idx, col_names.index(name))].set_facecolor(_status_color(status))
        if metric_col is not None:
            metric_value = str(original_row.get("Metric", ""))
            if metric_value in key_metric_names or metric_value.startswith("Actual objective"):
                for col_idx in range(cols):
                    table[(row_idx, col_idx)].set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _make_key_tables(table: pd.DataFrame, report: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = table[[
        "Configuration",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "P_minus_R_kw_per_server",
        "P_plus_R_kw_per_server",
        "Weights",
    ]].copy()

    focus_metrics = [
        "Actual objective",
        "Predicted objective",
        "Actual scaled cost_power",
        "Actual scaled cost_error",
        "Actual scaled cost_qos",
        "FlexDC full objective",
        "FlexDC RSR total cost",
        "FlexDC Ctrack weighted cost",
        "FlexDC CQoS SoftPlus cost",
        "90th percentile tracking error",
        "Mean abs normalized tracking error",
        "QoS violation ratio",
    ]
    focus = report[report["Metric"].map(lambda metric: any(str(metric).startswith(x) for x in focus_metrics))].copy()

    unscaled_metrics = [
        "Predicted raw cost_power",
        "Actual raw cost_power",
        "Predicted raw cost_error",
        "Actual raw cost_error",
        "Predicted raw cost_qos",
        "Actual raw cost_qos",
        "Predicted raw component sum",
        "Actual raw component sum",
    ]
    unscaled = report[report["Metric"].isin(unscaled_metrics)].copy()

    objective_rows = report[report["Category"].isin(["CONDOR objective", "CONDOR scaled components", "CONDOR unscaled components", "FlexDC objective", "Diagnostics"])].copy()
    return config, focus, unscaled, objective_rows

def write_report_outputs(table: pd.DataFrame, cost_weights: list[float], args: argparse.Namespace, out_dir: Path) -> dict[str, Path]:
    overview, report = make_report_tables(table, cost_weights, args)
    config_table, focus_report, unscaled_report, full_report_for_png = _make_key_tables(table, report)

    overview_path = out_dir / "end_to_end_validation_overview.csv"
    report_path = out_dir / "end_to_end_validation_report.csv"
    focus_path = out_dir / "end_to_end_validation_focus_report.csv"
    config_path = out_dir / "end_to_end_validation_config_table.csv"
    unscaled_path = out_dir / "end_to_end_validation_unscaled_costs.csv"
    markdown_path = out_dir / "end_to_end_validation_report.md"

    overview.to_csv(overview_path, index=False)
    report.to_csv(report_path, index=False)
    focus_report.to_csv(focus_path, index=False)
    config_table.to_csv(config_path, index=False)
    unscaled_report.to_csv(unscaled_path, index=False)
    write_report_markdown(markdown_path, overview, report, args, cost_weights)

    config_tag = _safe_tag(
        f"{Path(args.out_dir).name}_"
        f"{Path(args.workload_config).stem}_"
        f"N{args.server_count}_U{args.utilization}_"
        f"cw{'-'.join(str(x) for x in cost_weights)}"
    )
    title = (
        f"{Path(args.workload_config).stem} | N={args.server_count} | U={args.utilization} | "
        f"cost weights={cost_weights} | iterations={args.iterations} | lr={args.lr}"
    )
    overview_png = out_dir / f"001_{config_tag}_overview_table.png"
    focus_png = out_dir / f"002_{config_tag}_focus_metric_changes.png"
    config_png = out_dir / f"003_{config_tag}_configuration_and_weights.png"
    full_png = out_dir / f"004_{config_tag}_full_metric_change_table.png"
    unscaled_png = out_dir / f"005_{config_tag}_unscaled_cost_components.png"

    save_table_png(overview.round(6), overview_png, f"001 Key result overview | {title}", style="overview", max_col_width=34)
    save_table_png(focus_report, focus_png, f"002 Focus metrics: change from start to selected | {title}", style="report", max_col_width=38)
    save_table_png(config_table.round(6), config_png, f"003 Configuration and weights | {title}", style="overview", max_col_width=46)
    save_table_png(full_report_for_png, full_png, f"004 Full metric changes | {title}", style="report", max_col_width=38)
    save_table_png(unscaled_report, unscaled_png, f"005 Unscaled CONDOR costs | {title}", style="report", max_col_width=38)

    return {
        "overview_csv": overview_path,
        "report_csv": report_path,
        "focus_report_csv": focus_path,
        "config_table_csv": config_path,
        "unscaled_costs_csv": unscaled_path,
        "report_markdown": markdown_path,
        "overview_png": overview_png,
        "focus_png": focus_png,
        "config_png": config_png,
        "full_png": full_png,
        "unscaled_png": unscaled_png,
    }

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

    start_results, start_diagnostics = run_wizard(
        args,
        "start",
        candidate["starting_pbar_kw_per_server"],
        candidate["starting_r_kw_per_server"],
        candidate["starting_weights"],
    )
    opt_results, opt_diagnostics = run_wizard(
        args,
        "optimized",
        candidate["optimized_pbar_kw_per_server"],
        candidate["optimized_r_kw_per_server"],
        candidate["optimized_weights"],
    )
    table = make_validation_table(cost_weights, prediction_table, start_results, opt_results, start_diagnostics, opt_diagnostics)
    table.to_csv(out_dir / "end_to_end_validation_summary.csv", index=False)
    report_paths = write_report_outputs(table, cost_weights, args, out_dir)

    print("\nEnd-to-end validation summary")
    print(table.round(6).to_string(index=False))
    print(f"\nSaved: {out_dir / 'end_to_end_validation_summary.csv'}")
    print("\nGenerated report files")
    for path in report_paths.values():
        print(f"  {path}")

    if run is not None:
        for _, row in table.iterrows():
            prefix = "start" if row["Configuration"].startswith("Starting") else "selected"
            run.log({
                f"validation/{prefix}_predicted_objective": row["Predicted_Optimization_Objective"],
                f"validation/{prefix}_actual_objective": row["Actual_Optimization_Objective"],
                f"validation/{prefix}_tracking_error": row["MeanAbs_Normalized_Tracking_Error"],
                f"validation/{prefix}_tracking_error_p90": row["Ctrack_Epsilon_90th"],
                f"validation/{prefix}_qos_violation_ratio": row["QoS_Violation_Ratio"],
                f"validation/{prefix}_flexdc_full_objective": row["FlexDC_Full_Objective_Cost"],
            })
        try:
            import wandb
            run.log({
                "validation/overview_table": wandb.Table(dataframe=pd.read_csv(report_paths["overview_csv"])),
                "validation/metric_change_table": wandb.Table(dataframe=pd.read_csv(report_paths["report_csv"])),
            })
            for key in ("overview_png", "focus_png", "config_png", "full_png", "unscaled_png"):
                if key in report_paths and report_paths[key].exists():
                    run.log({f"validation/{key}": wandb.Image(str(report_paths[key]))})
            for path in report_paths.values():
                if path.exists():
                    run.save(str(path))
        except Exception as exc:  # W&B logging should not invalidate a completed validation.
            print(f"Warning: failed to log report tables/images to W&B: {exc}")
        run.finish()


if __name__ == "__main__":
    main()
