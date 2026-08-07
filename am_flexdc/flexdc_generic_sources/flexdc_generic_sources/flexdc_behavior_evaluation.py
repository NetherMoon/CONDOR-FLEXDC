"""Evaluation helpers for profile-holdout CONDOR-FlexDC experiments.

This module evaluates a frozen model on the J=2 profile-generalization test
suite. It reports direct behavior prediction, reconstructed objective quality,
constraint classification, decision-boundary behavior, ranking/top-k quality,
and simulator-seed stability. It never performs checkpoint selection.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRACKING_THRESHOLD = 0.30
QOS_THRESHOLD = 0.10


def safe_slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "table"


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _safe_r2(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    if len(actual_arr) < 2 or np.allclose(actual_arr, actual_arr[0]):
        return float("nan")
    return float(r2_score(actual_arr, predicted_arr))


def _safe_spearman(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_arr) & np.isfinite(predicted_arr)
    actual_arr = actual_arr[valid]
    predicted_arr = predicted_arr[valid]
    if len(actual_arr) < 2 or np.allclose(actual_arr, actual_arr[0]) or np.allclose(predicted_arr, predicted_arr[0]):
        return float("nan")
    correlation = spearmanr(actual_arr, predicted_arr).correlation
    return float(correlation) if correlation is not None else float("nan")


def classification_metrics(actual: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=bool)
    predicted_arr = np.asarray(predicted, dtype=bool)
    if len(actual_arr) != len(predicted_arr):
        raise ValueError("actual and predicted labels must have equal length")
    tp = int(np.sum(actual_arr & predicted_arr))
    fp = int(np.sum((~actual_arr) & predicted_arr))
    tn = int(np.sum((~actual_arr) & (~predicted_arr)))
    fn = int(np.sum(actual_arr & (~predicted_arr)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = (
        _safe_div(2.0 * precision * recall, precision + recall)
        if np.isfinite(precision) and np.isfinite(recall)
        else float("nan")
    )
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": _safe_div(tp + tn, len(actual_arr)),
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "False_Feasible_Rate_Among_Actual_Infeasible": _safe_div(fp, fp + tn),
        "False_Infeasible_Rate_Among_Actual_Feasible": _safe_div(fn, fn + tp),
        "Actual_Feasible": int(tp + fn),
        "Predicted_Feasible": int(tp + fp),
    }


def regression_metrics(actual: Sequence[float], predicted: Sequence[float], prefix: str) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_arr) & np.isfinite(predicted_arr)
    actual_arr = actual_arr[valid]
    predicted_arr = predicted_arr[valid]
    if len(actual_arr) == 0:
        return {
            f"{prefix}_MAE": float("nan"),
            f"{prefix}_RMSE": float("nan"),
            f"{prefix}_R2": float("nan"),
            f"{prefix}_Spearman": float("nan"),
            f"{prefix}_Signed_Bias": float("nan"),
        }
    return {
        f"{prefix}_MAE": float(mean_absolute_error(actual_arr, predicted_arr)),
        f"{prefix}_RMSE": float(math.sqrt(mean_squared_error(actual_arr, predicted_arr))),
        f"{prefix}_R2": _safe_r2(actual_arr, predicted_arr),
        f"{prefix}_Spearman": _safe_spearman(actual_arr, predicted_arr),
        f"{prefix}_Signed_Bias": float(np.mean(predicted_arr - actual_arr)),
    }


def enrich_prediction_rows(predictions: pd.DataFrame, source_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach profile-holdout metadata and base IDs to model predictions."""
    predictions = predictions.copy()
    source_rows = source_rows.copy()
    if "Plan_Row_ID" not in predictions.columns or "Plan_Row_ID" not in source_rows.columns:
        raise KeyError("Plan_Row_ID is required in predictions and source rows.")
    predictions["Plan_Row_ID"] = predictions["Plan_Row_ID"].astype(str)
    source_rows["Plan_Row_ID"] = source_rows["Plan_Row_ID"].astype(str)
    if source_rows["Plan_Row_ID"].duplicated().any():
        raise ValueError("Source test rows contain duplicate Plan_Row_ID values.")

    preferred_metadata = [
        "Plan_Row_ID",
        "Base_Plan_Row_ID",
        "Data_Split",
        "Benchmark_Type",
        "Model_ID",
        "Model_Display_Name",
        "Split_Type",
        "Workload_Type",
        "Job_Profiles",
        "Job_Profile_1",
        "Job_Profile_2",
        "Heldout_Profile_Count",
        "Heldout_Profiles",
        "Visible_Profiles_In_Workload",
        "Profiles_Visible_During_Training",
        "Profiles_Completely_Held_Out",
        "Simulation_Seed",
        "simulation_seed",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "weights",
        "pr_id",
        "weight_id",
    ]
    metadata_columns = [column for column in preferred_metadata if column in source_rows.columns]
    metadata = source_rows[metadata_columns].copy()
    if "Simulation_Seed" not in metadata.columns and "simulation_seed" in metadata.columns:
        metadata = metadata.rename(columns={"simulation_seed": "Simulation_Seed"})

    overlapping = [column for column in metadata.columns if column in predictions.columns and column != "Plan_Row_ID"]
    metadata = metadata.drop(columns=overlapping)
    enriched = predictions.merge(metadata, on="Plan_Row_ID", how="left", validate="one_to_one")
    if enriched["Base_Plan_Row_ID"].isna().any() if "Base_Plan_Row_ID" in enriched.columns else True:
        raise ValueError("Could not attach Base_Plan_Row_ID to every prediction row.")
    if "Benchmark_Type" not in enriched.columns:
        raise KeyError("Benchmark_Type is required for profile-holdout evaluation.")
    return enriched


def summarize_subset(frame: pd.DataFrame, label: str) -> dict[str, object]:
    row: dict[str, object] = {
        "Subset": label,
        "Rows": int(len(frame)),
        "Base_Groups": int(frame["Base_Plan_Row_ID"].astype(str).nunique()) if len(frame) else 0,
        "Workloads": int(frame["Workload_Name"].astype(str).nunique()) if len(frame) else 0,
        "Contexts": int(frame["Context_ID"].astype(str).nunique()) if len(frame) else 0,
    }
    if len(frame) == 0:
        return row
    row.update(regression_metrics(frame["Actual_Mean_Tracking"], frame["Predicted_Mean_Tracking"], "Mean_Tracking"))
    row.update(regression_metrics(frame["Actual_P90_Tracking"], frame["Predicted_P90_Tracking"], "P90_Tracking"))
    row.update(regression_metrics(frame["Actual_Max_Pj"], frame["Predicted_Max_Pj"], "Max_Pj"))
    row.update(regression_metrics(frame["Actual_M_RSR"], frame["Predicted_M_RSR"], "M_RSR"))
    row.update(regression_metrics(frame["Actual_Full_Objective"], frame["Predicted_Full_Objective"], "Full_Objective"))

    for name, actual_col, predicted_col in [
        ("Tracking", "Actual_Tracking_Pass", "Predicted_Tracking_Pass"),
        ("QoS", "Actual_QoS_Pass", "Predicted_QoS_Pass"),
        ("Combined", "Actual_Both_Pass", "Predicted_Both_Pass"),
    ]:
        metrics = classification_metrics(frame[actual_col].astype(bool), frame[predicted_col].astype(bool))
        row.update({f"{name}_{key}": value for key, value in metrics.items()})
    return row


def metrics_by_group(frame: pd.DataFrame, group_columns: Sequence[str], label_prefix: str | None = None) -> pd.DataFrame:
    columns = [column for column in group_columns if column in frame.columns]
    if not columns:
        return pd.DataFrame([summarize_subset(frame, label_prefix or "All Test Rows")])
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(columns, dropna=False, sort=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        label = " | ".join(f"{column}={value}" for column, value in zip(columns, keys_tuple))
        record = {column: value for column, value in zip(columns, keys_tuple)}
        record.update(summarize_subset(group, label_prefix + " | " + label if label_prefix else label))
        rows.append(record)
    return pd.DataFrame(rows)


def boundary_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        (
            "Tracking decision band: actual p90 in [0.20, 0.40]",
            frame["Actual_P90_Tracking"].between(0.20, 0.40),
        ),
        (
            "QoS decision band: actual max Pj in [0.05, 0.15]",
            frame["Actual_Max_Pj"].between(0.05, 0.15),
        ),
        (
            "Either decision band",
            frame["Actual_P90_Tracking"].between(0.20, 0.40)
            | frame["Actual_Max_Pj"].between(0.05, 0.15),
        ),
    ]
    rows: list[dict[str, object]] = []
    for label, mask in definitions:
        record = summarize_subset(frame.loc[mask].copy(), label)
        if int(mask.sum()):
            record["Tracking_Constraint_Flip_Rate"] = float(
                np.mean(
                    frame.loc[mask, "Actual_Tracking_Pass"].astype(bool).to_numpy()
                    != frame.loc[mask, "Predicted_Tracking_Pass"].astype(bool).to_numpy()
                )
            )
            record["QoS_Constraint_Flip_Rate"] = float(
                np.mean(
                    frame.loc[mask, "Actual_QoS_Pass"].astype(bool).to_numpy()
                    != frame.loc[mask, "Predicted_QoS_Pass"].astype(bool).to_numpy()
                )
            )
        rows.append(record)
    return pd.DataFrame(rows)


def failure_type_table(frame: pd.DataFrame) -> pd.DataFrame:
    actual_tracking = frame["Actual_Tracking_Pass"].astype(bool)
    actual_qos = frame["Actual_QoS_Pass"].astype(bool)
    predicted_tracking = frame["Predicted_Tracking_Pass"].astype(bool)
    predicted_qos = frame["Predicted_QoS_Pass"].astype(bool)

    def category(track: pd.Series, qos: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [track & qos, (~track) & qos, track & (~qos), (~track) & (~qos)],
                ["Feasible", "Tracking-Only Failure", "QoS-Only Failure", "Both Fail"],
                default="Unknown",
            ),
            index=frame.index,
        )

    data = frame[[column for column in ["Benchmark_Type", "Workload_Type"] if column in frame.columns]].copy()
    data["Actual_Failure_Type"] = category(actual_tracking, actual_qos)
    data["Predicted_Failure_Type"] = category(predicted_tracking, predicted_qos)
    group_columns = [column for column in ["Benchmark_Type", "Workload_Type"] if column in data.columns]
    return (
        data.groupby(group_columns + ["Actual_Failure_Type", "Predicted_Failure_Type"], dropna=False)
        .size()
        .rename("Rows")
        .reset_index()
        .sort_values(group_columns + ["Actual_Failure_Type", "Predicted_Failure_Type"])
        .reset_index(drop=True)
    )


def _aggregate_base_groups(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["Base_Plan_Row_ID"]
    identity_columns = [
        "Workload_Name",
        "Context_ID",
        "Benchmark_Type",
        "Workload_Type",
        "Heldout_Profile_Count",
        "Heldout_Profiles",
    ]
    aggregation: dict[str, str] = {
        column: "first" for column in identity_columns if column in frame.columns
    }
    for column in [
        "Actual_Full_Objective",
        "Predicted_Full_Objective",
        "Actual_P90_Tracking",
        "Predicted_P90_Tracking",
        "Actual_Max_Pj",
        "Predicted_Max_Pj",
    ]:
        aggregation[column] = "mean"
    for column in [
        "Actual_Tracking_Pass",
        "Predicted_Tracking_Pass",
        "Actual_QoS_Pass",
        "Predicted_QoS_Pass",
        "Actual_Both_Pass",
        "Predicted_Both_Pass",
    ]:
        aggregation[column] = "mean"
    grouped = frame.groupby(group_columns, sort=False).agg(aggregation).reset_index()
    for column in [
        "Actual_Tracking_Pass",
        "Predicted_Tracking_Pass",
        "Actual_QoS_Pass",
        "Predicted_QoS_Pass",
        "Actual_Both_Pass",
        "Predicted_Both_Pass",
    ]:
        grouped[column] = grouped[column] >= 0.5
    return grouped


def _top_k_overlap(frame: pd.DataFrame, k: int, actual_col: str, predicted_col: str) -> float:
    if len(frame) == 0:
        return float("nan")
    k = min(int(k), len(frame))
    actual_ids = set(frame.nsmallest(k, actual_col)["Base_Plan_Row_ID"].astype(str))
    predicted_ids = set(frame.nsmallest(k, predicted_col)["Base_Plan_Row_ID"].astype(str))
    return _safe_div(len(actual_ids & predicted_ids), k)


def ranking_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure constrained candidate-ranking quality by complete base group.

    Top-k overlap is computed *within the actual-feasible domain* and the
    predicted-feasible domain rather than ranking infeasible rows as though they
    were valid optimizer destinations.  Regret is reported only when the
    surrogate-selected candidate is actually feasible.  An infeasible selected
    point therefore yields NaN regret instead of a misleading negative value.
    """
    base = _aggregate_base_groups(frame)
    group_columns = [column for column in ["Benchmark_Type", "Workload_Name", "Context_ID"] if column in base.columns]
    rows: list[dict[str, object]] = []
    for keys, group in base.groupby(group_columns, dropna=False, sort=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        record: dict[str, object] = {column: value for column, value in zip(group_columns, keys_tuple)}
        predicted_feasible = group[group["Predicted_Both_Pass"]].copy()
        actual_feasible = group[group["Actual_Both_Pass"]].copy()
        common_feasible = group[group["Predicted_Both_Pass"] & group["Actual_Both_Pass"]].copy()

        def constrained_overlap(k: int) -> float:
            if len(actual_feasible) == 0 or len(predicted_feasible) == 0:
                return float("nan")
            actual_ids = set(
                actual_feasible.nsmallest(min(k, len(actual_feasible)), "Actual_Full_Objective")["Base_Plan_Row_ID"].astype(str)
            )
            predicted_ids = set(
                predicted_feasible.nsmallest(min(k, len(predicted_feasible)), "Predicted_Full_Objective")["Base_Plan_Row_ID"].astype(str)
            )
            denominator = min(k, len(actual_ids), len(predicted_ids))
            return float(len(actual_ids & predicted_ids) / denominator) if denominator else float("nan")

        record.update({
            "Base_Groups": int(len(group)),
            "Actual_Feasible_Groups": int(len(actual_feasible)),
            "Predicted_Feasible_Groups": int(len(predicted_feasible)),
            "Common_Feasible_Groups": int(len(common_feasible)),
            "Objective_Spearman_All": _safe_spearman(group["Actual_Full_Objective"], group["Predicted_Full_Objective"]),
            "Objective_Spearman_Actual_Feasible": (
                _safe_spearman(actual_feasible["Actual_Full_Objective"], actual_feasible["Predicted_Full_Objective"])
                if len(actual_feasible) >= 2 else float("nan")
            ),
            "Constrained_Top5_Overlap": constrained_overlap(5),
            "Constrained_Top10_Overlap": constrained_overlap(10),
            # Backward-compatible aliases now use the constrained definition.
            "Objective_Spearman": _safe_spearman(group["Actual_Full_Objective"], group["Predicted_Full_Objective"]),
            "Top5_Overlap": constrained_overlap(5),
            "Top10_Overlap": constrained_overlap(10),
        })
        predicted_feasible = predicted_feasible.sort_values("Predicted_Full_Objective")
        actual_feasible = actual_feasible.sort_values("Actual_Full_Objective")
        if len(predicted_feasible):
            chosen = predicted_feasible.iloc[0]
            selected_actually_feasible = bool(chosen["Actual_Both_Pass"])
            record["Selected_Base_Plan_Row_ID"] = str(chosen["Base_Plan_Row_ID"])
            record["Selected_Actually_Feasible"] = selected_actually_feasible
            record["Selected_Actual_Objective"] = float(chosen["Actual_Full_Objective"])
            record["Selected_Predicted_Objective"] = float(chosen["Predicted_Full_Objective"])
            record["Selection_Status"] = "actual_feasible" if selected_actually_feasible else "actual_infeasible"
        else:
            record["Selected_Base_Plan_Row_ID"] = None
            record["Selected_Actually_Feasible"] = False
            record["Selected_Actual_Objective"] = float("nan")
            record["Selected_Predicted_Objective"] = float("nan")
            record["Selection_Status"] = "no_predicted_feasible_candidate"
        if len(actual_feasible):
            best = float(actual_feasible.iloc[0]["Actual_Full_Objective"])
            record["Best_Known_Actual_Feasible_Objective"] = best
            selected = float(record["Selected_Actual_Objective"])
            record["Regret_Percent"] = (
                float((selected - best) / abs(best) * 100.0)
                if bool(record["Selected_Actually_Feasible"]) and np.isfinite(selected) and best != 0
                else float("nan")
            )
            record["Regret_Status"] = (
                "computed" if np.isfinite(record["Regret_Percent"])
                else "not_computed_selected_candidate_infeasible_or_missing"
            )
        else:
            record["Best_Known_Actual_Feasible_Objective"] = float("nan")
            record["Regret_Percent"] = float("nan")
            record["Regret_Status"] = "not_computed_no_actual_feasible_candidate"
        rows.append(record)
    return pd.DataFrame(rows)


def seed_stability_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for base_id, group in frame.groupby("Base_Plan_Row_ID", sort=False):
        if len(group) < 2:
            continue
        actual_labels = group["Actual_Both_Pass"].astype(bool)
        predicted_labels = group["Predicted_Both_Pass"].astype(bool)
        record: dict[str, object] = {
            "Base_Plan_Row_ID": str(base_id),
            "Runs": int(len(group)),
            "Workload_Name": str(group["Workload_Name"].iloc[0]),
            "Benchmark_Type": str(group["Benchmark_Type"].iloc[0]),
            "Actual_P90_Range": float(group["Actual_P90_Tracking"].max() - group["Actual_P90_Tracking"].min()),
            "Predicted_P90_Range": float(group["Predicted_P90_Tracking"].max() - group["Predicted_P90_Tracking"].min()),
            "Actual_Max_Pj_Range": float(group["Actual_Max_Pj"].max() - group["Actual_Max_Pj"].min()),
            "Predicted_Max_Pj_Range": float(group["Predicted_Max_Pj"].max() - group["Predicted_Max_Pj"].min()),
            "Actual_Objective_Range": float(group["Actual_Full_Objective"].max() - group["Actual_Full_Objective"].min()),
            "Actual_Feasibility_Label_Flip": bool(actual_labels.nunique() > 1),
            "Predicted_Feasibility_Label_Flip": bool(predicted_labels.nunique() > 1),
        }
        if "Simulation_Seed" in group.columns:
            record["Simulation_Seeds"] = json.dumps(
                sorted(pd.to_numeric(group["Simulation_Seed"], errors="coerce").dropna().astype(int).unique().tolist())
            )
        rows.append(record)
    return pd.DataFrame(rows)


def run_profile_holdout_evaluation(
    predictions: pd.DataFrame,
    source_test_rows: pd.DataFrame,
) -> dict[str, pd.DataFrame | dict]:
    enriched = enrich_prediction_rows(predictions, source_test_rows)
    overall = pd.DataFrame([summarize_subset(enriched, "All Test Rows")])
    by_benchmark = metrics_by_group(enriched, ["Benchmark_Type"])
    by_benchmark_workload_type = metrics_by_group(enriched, ["Benchmark_Type", "Workload_Type"])
    by_workload = metrics_by_group(enriched, ["Benchmark_Type", "Workload_Name"])
    by_context = metrics_by_group(enriched, ["Benchmark_Type", "Context_ID"])
    by_heldout_profiles = metrics_by_group(enriched, ["Benchmark_Type", "Heldout_Profiles"])
    boundaries = boundary_metrics(enriched)
    failures = failure_type_table(enriched)
    rankings = ranking_metrics(enriched)
    seed_stability = seed_stability_metrics(enriched)

    compact_summary = {
        "rows": int(len(enriched)),
        "base_groups": int(enriched["Base_Plan_Row_ID"].nunique()),
        "workloads": int(enriched["Workload_Name"].nunique()),
        "benchmark_types": sorted(enriched["Benchmark_Type"].astype(str).unique().tolist()),
        "actual_feasible_rows": int(enriched["Actual_Both_Pass"].astype(bool).sum()),
        "predicted_feasible_rows": int(enriched["Predicted_Both_Pass"].astype(bool).sum()),
        "simulator_repeat_groups": int((enriched.groupby("Base_Plan_Row_ID").size() > 1).sum()),
    }
    return {
        "predictions_enriched": enriched,
        "overall_metrics": overall,
        "metrics_by_benchmark": by_benchmark,
        "metrics_by_benchmark_workload_type": by_benchmark_workload_type,
        "metrics_by_workload": by_workload,
        "metrics_by_context": by_context,
        "metrics_by_heldout_profiles": by_heldout_profiles,
        "boundary_metrics": boundaries,
        "failure_type_confusion": failures,
        "ranking_metrics": rankings,
        "seed_stability": seed_stability,
        "summary": compact_summary,
    }


def write_evaluation_outputs(
    outputs: Mapping[str, pd.DataFrame | dict],
    output_dir: str | Path,
    *,
    prefix: str,
    write_html: bool = True,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, value in outputs.items():
        safe_name = safe_slug(name)
        if isinstance(value, pd.DataFrame):
            csv_path = output_dir / f"{prefix}_{safe_name}.csv"
            value.to_csv(csv_path, index=False)
            written.append(csv_path)
            if write_html:
                # Write a complete standalone report instead of a bare table
                # fragment.  The earlier generic bundle regressed to raw
                # DataFrame HTML, while the original notebooks produced
                # presentation-ready pages with embedded CSS.
                from flexdc_presentation import build_report

                html_path = output_dir / f"{prefix}_{safe_name}.html"
                html_path.write_text(
                    build_report(
                        title=f"FlexDC Evaluation: {name.replace('_', ' ').title()}",
                        subtitle=f"Run prefix: {prefix}",
                        tables={name.replace('_', ' ').title(): value},
                    ),
                    encoding="utf-8",
                )
                written.append(html_path)
        else:
            json_path = output_dir / f"{prefix}_{safe_name}.json"
            json_path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
            written.append(json_path)
    return written


def styled_table(frame: pd.DataFrame, caption: str | None = None, precision: int = 5):
    """Return a clean Pandas Styler for Colab/Jupyter display."""
    if frame is None:
        frame = pd.DataFrame()
    styler = (
        frame.style.hide(axis="index")
        .format(precision=precision, na_rep="—")
        .set_properties(
            **{
                "text-align": "left",
                "white-space": "normal",
                "font-size": "12px",
                "padding": "7px",
                "border": "1px solid #cbd5e1",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#111827"),
                        ("color", "#ffffff"),
                        ("font-weight", "700"),
                        ("text-align", "left"),
                        ("padding", "8px"),
                    ],
                },
                {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
                {
                    "selector": "caption",
                    "props": [
                        ("caption-side", "top"),
                        ("font-size", "17px"),
                        ("font-weight", "700"),
                        ("text-align", "left"),
                        ("padding", "0 0 8px 0"),
                    ],
                },
            ]
        )
    )
    return styler.set_caption(caption) if caption else styler


def render_styled_table(frame: pd.DataFrame, caption: str | None = None, precision: int = 5):
    """Render a Styler as real HTML in Colab/Jupyter.

    Calling ``display(styler)`` printed raw CSS in some Colab runtimes.  This
    explicit conversion preserves the original presentation layer while
    preventing that failure mode.
    """
    from IPython.display import HTML, display
    styler = styled_table(frame, caption=caption, precision=precision)
    display(HTML(styler.to_html()))
    return styler


def log_evaluation_to_wandb(
    run,
    outputs: Mapping[str, pd.DataFrame | dict],
    *,
    namespace: str = "profile_holdout",
    max_table_rows: int = 5000,
) -> None:
    if run is None:
        return
    import wandb

    payload = {}
    summary = outputs.get("summary", {})
    if isinstance(summary, Mapping):
        for key, value in summary.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                run.summary[f"{namespace}/{key}"] = float(value)
    for name, value in outputs.items():
        if not isinstance(value, pd.DataFrame) or len(value) == 0:
            continue
        sample = value.head(max_table_rows).copy()
        payload[f"{namespace}/{name}"] = wandb.Table(dataframe=sample)
    if payload:
        run.log(payload)
