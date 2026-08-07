#!/usr/bin/env python3
"""Combine, audit, and split exact-plan FlexDC outputs for behavior-model training.

Run this script from the FlexDC repository root. It replaces the old
version-specific combine and vet scripts with one reusable pipeline:

1. discover exact-plan worker results and diagnostics;
2. verify unique and exact Plan_Row_ID coverage against the plan;
3. merge authoritative plan metadata with simulator labels;
4. collapse Weight_0...Weight_J columns and reconstruct workload metadata when
   the worker output omitted it;
5. validate the schema required by the generic CONDOR behavior model;
6. create strict profile-holdout datasets for any or all of the four focused
   J=2 model specifications;
7. write audits, manifests, summaries, and one Colab-ready ZIP.

No dataset-size, worker-count, model-version, or sweep-density constants are
hard-coded. The exact plan is the source of truth.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "flexdc_generic_sources"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from flexdc_profile_split import MODEL_SPECS, assign_profile_holdout_split, resolve_model_spec

RAW_NAME = "grid_search_results.csv"
DIAGNOSTIC_NAME = "grid_search_diagnostics.csv"
TRACKING_THRESHOLD = 0.30
QOS_THRESHOLD = 0.10
TOL = 1e-8

TRAINING_REQUIRED_COLUMNS = [
    "Workload_Name",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "Pbar_ratio",
    "R_ratio",
    "server_count",
    "utilization",
    "workload_mix_size",
    "workload_mix",
    "weights",
    "Mtrack_Error_MeanAbs_Normalized",
    "Ctrack_Epsilon_90th",
    "QoS_Delay_Probabilities",
    "Simulator_Power_Cost",
    "Simulator_RSR_Total_Cost",
    "Diagnostic_FullPaperObjective_Cost",
    "Mtrack_Price_Coefficient_piE",
    "Mtrack_Hour_Seconds",
    "R_actual_watts",
]


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_vector(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    text = str(value).strip()
    text = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
    try:
        parsed = json.loads(text)
    except Exception:
        import ast
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        raise ValueError(f"Expected vector, got {value!r}")
    return [float(x) for x in parsed]


def collapse_weight_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if "weights" in frame.columns:
        frame = frame.copy()
        frame["weights"] = frame["weights"].map(lambda value: json.dumps(parse_vector(value)))
        return frame
    columns = sorted(
        [column for column in frame.columns if re.fullmatch(r"Weight_\d+", str(column))],
        key=lambda column: int(str(column).split("_")[1]),
    )
    if not columns:
        return frame
    first_index = frame.columns.get_loc(columns[0])
    weights = frame[columns].apply(
        lambda row: json.dumps([float(row[column]) for column in columns if not pd.isna(row[column])]),
        axis=1,
    )
    output = frame.drop(columns=columns)
    output.insert(first_index, "weights", weights)
    return output


def collect_from_manifest(path: Path) -> tuple[list[Path], list[Path]]:
    manifest = pd.read_csv(path, low_memory=False)
    if "status" in manifest.columns:
        allowed = {"completed", "skipped_complete", "complete"}
        bad = ~manifest["status"].astype(str).str.lower().isin(allowed)
        if bad.any():
            counts = manifest.loc[bad, "status"].astype(str).value_counts().to_dict()
            raise AuditError(f"Launch manifest contains incomplete workers: {counts}")
        manifest = manifest.loc[~bad].copy()
    required = {"results_csv", "diagnostics_csv"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Launch manifest is missing: {sorted(missing)}")
    results = [Path(value) for value in manifest["results_csv"].dropna().astype(str)]
    diagnostics = [Path(value) for value in manifest["diagnostics_csv"].dropna().astype(str)]
    return results, diagnostics


def collect_from_scan(optimization_dir: Path, prefix: str) -> tuple[list[Path], list[Path]]:
    results = sorted(optimization_dir.glob(f"{prefix}*/{RAW_NAME}"))
    diagnostics = sorted(optimization_dir.glob(f"{prefix}*/{DIAGNOSTIC_NAME}"))
    if not results:
        raise FileNotFoundError(f"No {RAW_NAME} files under {optimization_dir} matching {prefix}*")
    if not diagnostics:
        raise FileNotFoundError(f"No {DIAGNOSTIC_NAME} files under {optimization_dir} matching {prefix}*")
    return results, diagnostics


def resolve_paths(paths: Sequence[Path], repo_root: Path) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        resolved.append(candidate)
    return resolved


def read_and_combine(paths: Sequence[Path], kind: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame.insert(0, "Source_Output_Dir", path.parent.name)
        frame.insert(1, "Source_CSV_Path", str(path))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = collapse_weight_columns(combined)
    plan_id = find_column(combined, "Plan_Row_ID", "plan_row_id")
    if plan_id is None:
        raise AuditError(f"Combined {kind} rows do not contain Plan_Row_ID.")
    if plan_id != "Plan_Row_ID":
        combined["Plan_Row_ID"] = combined[plan_id].astype(str)
    else:
        combined["Plan_Row_ID"] = combined["Plan_Row_ID"].astype(str)
    duplicates = combined["Plan_Row_ID"].duplicated(keep=False)
    if duplicates.any():
        examples = combined.loc[duplicates, ["Plan_Row_ID", "Source_Output_Dir"]].head(20)
        raise AuditError(f"Duplicate Plan_Row_ID values in {kind}:\n{examples.to_string(index=False)}")
    return combined


def find_column(frame: pd.DataFrame, *aliases: str) -> str | None:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def ensure_column(frame: pd.DataFrame, canonical: str, aliases: Iterable[str], required: bool = False) -> None:
    if canonical in frame.columns:
        return
    source = find_column(frame, *aliases)
    if source is not None:
        frame[canonical] = frame[source]
    elif required:
        raise KeyError(f"Missing required column {canonical}; accepted aliases={list(aliases)}")


def canonicalize_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    alias_map = {
        "Plan_Row_ID": ["plan_row_id"],
        "Base_Plan_Row_ID": ["base_plan_row_id", "Base_Row_ID"],
        "Workload_Name": ["workload_name"],
        "Pbar_kw_per_server": ["pbar_kw_per_server", "Pbar", "Pbar_kW_per_server"],
        "R_kw_per_server": ["r_kw_per_server", "R", "R_kW_per_server"],
        "server_count": ["Server_Count", "N"],
        "utilization": ["Utilization", "U"],
        "Simulation_Seed": ["simulation_seed", "random_seed"],
        "workload_mix_size": ["Workload_Mix_Size", "job_type_count", "J"],
        "workload_mix": ["Workload_Mix"],
        "weights": ["Weights"],
        "Pbar_ratio": ["Pbar_Ratio", "pbar_ratio"],
        "R_ratio": ["R_Ratio", "r_ratio"],
        "Mtrack_Error_MeanAbs_Normalized": ["Mean_Tracking_Error_Normalized"],
        "Ctrack_Epsilon_90th": ["Tracking_Error_90th", "p90_tracking"],
        "QoS_Delay_Probabilities": ["QOS_Delay_Probabilities", "QoS_Probabilities"],
        "Simulator_Power_Cost": ["Mpower_Cost", "Power_Cost"],
        "Simulator_RSR_Total_Cost": ["M_RSR", "RSR_Total_Cost"],
        "Diagnostic_FullPaperObjective_Cost": ["FullPaperObjective_Cost", "Full_Objective"],
        "Mtrack_Price_Coefficient_piE": ["Tracking_Price_Coefficient", "piE"],
        "Mtrack_Hour_Seconds": ["Hour_Seconds", "simulation_duration_seconds"],
        "R_actual_watts": ["R_Actual_Watts"],
        "Ctrack_Weighted_Cost": ["Tracking_Weighted_Cost"],
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost": ["FlexDC_SoftPlus_QoS_Cost", "CQoS"],
    }
    for canonical, aliases in alias_map.items():
        ensure_column(frame, canonical, aliases, required=False)
    if "weights" in frame.columns:
        frame["weights"] = frame["weights"].map(lambda value: json.dumps(parse_vector(value)))
    return frame


def resolve_repo_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def read_workload_ini(path: Path) -> tuple[list[str], np.ndarray]:
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(path)
    default_size = parser.defaults().get("job_size", "1")
    names: list[str] = []
    rows: list[list[float]] = []
    for section in parser.sections():
        job = parser[section]
        names.append(section)
        rows.append([
            job.getfloat("min_job_power_watts"),
            job.getfloat("max_job_power_watts"),
            job.getfloat("min_time_seconds"),
            job.getfloat("max_time_seconds"),
            job.getfloat("qos_constraint"),
            float(job.get("job_size", default_size)),
        ])
    if not rows:
        raise AuditError(f"No job sections found in {path}")
    return names, np.asarray(rows, dtype=float)


def read_experiment_ini(path: Path) -> dict[str, float]:
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(path)
    if "system" not in parser:
        raise AuditError(f"Experiment config lacks [system]: {path}")
    system = parser["system"]
    return {
        "idle_watts": system.getfloat("idle_watts"),
        "simulation_duration": system.getfloat("simulation_duration"),
    }


def add_config_derived_metadata(frame: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    frame = frame.copy()
    workload_path_col = find_column(frame, "workload_config", "Workload_Config", "workload_config_path")
    experiment_path_col = find_column(frame, "experiment_config", "Experiment_Config", "experiment_config_path")

    workload_cache: dict[str, tuple[list[str], np.ndarray]] = {}
    experiment_cache: dict[str, dict[str, float]] = {}

    def workload_data(value: Any) -> tuple[list[str], np.ndarray]:
        key = str(value)
        if key not in workload_cache:
            workload_cache[key] = read_workload_ini(resolve_repo_path(value, repo_root))
        return workload_cache[key]

    if workload_path_col is not None:
        if "Workload_Name" not in frame.columns or frame["Workload_Name"].isna().any():
            derived = frame[workload_path_col].map(lambda value: Path(str(value)).stem)
            if "Workload_Name" not in frame.columns:
                frame["Workload_Name"] = derived
            else:
                frame["Workload_Name"] = frame["Workload_Name"].fillna(derived)
        if "workload_mix" not in frame.columns or frame["workload_mix"].isna().any():
            derived = frame[workload_path_col].map(
                lambda value: json.dumps(workload_data(value)[1].tolist())
            )
            if "workload_mix" not in frame.columns:
                frame["workload_mix"] = derived
            else:
                frame["workload_mix"] = frame["workload_mix"].fillna(derived)
        if "workload_mix_size" not in frame.columns or frame["workload_mix_size"].isna().any():
            derived = frame[workload_path_col].map(lambda value: int(workload_data(value)[1].shape[0]))
            if "workload_mix_size" not in frame.columns:
                frame["workload_mix_size"] = derived
            else:
                frame["workload_mix_size"] = frame["workload_mix_size"].fillna(derived)
        if "Job_Section_Names" not in frame.columns:
            frame["Job_Section_Names"] = frame[workload_path_col].map(
                lambda value: json.dumps(workload_data(value)[0])
            )

    if "R_actual_watts" not in frame.columns and {"R_kw_per_server", "server_count"}.issubset(frame.columns):
        frame["R_actual_watts"] = (
            pd.to_numeric(frame["R_kw_per_server"], errors="raise")
            * 1000.0
            * pd.to_numeric(frame["server_count"], errors="raise")
        )

    need_ratios = "Pbar_ratio" not in frame.columns or "R_ratio" not in frame.columns
    need_duration = "Mtrack_Hour_Seconds" not in frame.columns
    if (need_ratios or need_duration) and workload_path_col is not None and experiment_path_col is not None:
        def experiment_data(value: Any) -> dict[str, float]:
            key = str(value)
            if key not in experiment_cache:
                experiment_cache[key] = read_experiment_ini(resolve_repo_path(value, repo_root))
            return experiment_cache[key]

        pbar_ratios: list[float] = []
        r_ratios: list[float] = []
        durations: list[float] = []
        for _, row in frame.iterrows():
            mix = workload_data(row[workload_path_col])[1]
            experiment = experiment_data(row[experiment_path_col])
            n = float(row["server_count"])
            u = float(row["utilization"])
            avg_max_power = float(np.mean(mix[:, 1]))
            pbar_denominator = avg_max_power * n * u + experiment["idle_watts"] * n * (1.0 - u)
            r_denominator = (avg_max_power * n - experiment["idle_watts"] * n) / 2.0
            p_actual = float(row["Pbar_kw_per_server"]) * 1000.0 * n
            r_actual = float(row["R_kw_per_server"]) * 1000.0 * n
            if pbar_denominator <= 0 or r_denominator <= 0:
                raise AuditError("Invalid P/R normalization denominator.")
            pbar_ratios.append(p_actual / pbar_denominator)
            r_ratios.append(r_actual / r_denominator)
            durations.append(experiment["simulation_duration"])
        if "Pbar_ratio" not in frame.columns:
            frame["Pbar_ratio"] = pbar_ratios
        if "R_ratio" not in frame.columns:
            frame["R_ratio"] = r_ratios
        if "Mtrack_Hour_Seconds" not in frame.columns:
            frame["Mtrack_Hour_Seconds"] = durations
    return frame


def compare_coverage(plan: pd.DataFrame, frame: pd.DataFrame, label: str) -> None:
    planned = set(plan["Plan_Row_ID"].astype(str))
    actual = set(frame["Plan_Row_ID"].astype(str))
    missing = sorted(planned - actual)
    extra = sorted(actual - planned)
    if missing or extra:
        raise AuditError(
            f"{label} Plan_Row_ID coverage mismatch: missing={len(missing)}, extra={len(extra)}, "
            f"missing examples={missing[:10]}, extra examples={extra[:10]}"
        )


def merge_outputs_with_plan(plan: pd.DataFrame, results: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    compare_coverage(plan, results, "results")
    compare_coverage(plan, diagnostics, "diagnostics")

    extra_diag = [
        column for column in diagnostics.columns
        if column == "Plan_Row_ID" or column not in results.columns
    ]
    merged = results.merge(
        diagnostics[extra_diag], on="Plan_Row_ID", how="inner", validate="one_to_one"
    )
    extra_plan = [
        column for column in plan.columns
        if column == "Plan_Row_ID" or column not in merged.columns
    ]
    merged = merged.merge(
        plan[extra_plan], on="Plan_Row_ID", how="inner", validate="one_to_one"
    )
    if len(merged) != len(plan):
        raise AuditError(f"Merged rows={len(merged)}, plan rows={len(plan)}")
    return merged


def validate_vectors(frame: pd.DataFrame) -> dict[str, Any]:
    bad_weights: list[str] = []
    bad_mix: list[str] = []
    bad_qos: list[str] = []
    job_counts: set[int] = set()
    for _, row in frame.iterrows():
        row_id = str(row["Plan_Row_ID"])
        weights = parse_vector(row["weights"])
        qos = parse_vector(row["QoS_Delay_Probabilities"])
        try:
            mix = np.asarray(json.loads(str(row["workload_mix"])), dtype=float)
        except Exception:
            import ast
            mix = np.asarray(ast.literal_eval(str(row["workload_mix"])), dtype=float)
        if mix.ndim != 2 or mix.shape[1] not in {6, 7}:
            bad_mix.append(row_id)
            continue
        j = int(mix.shape[0])
        job_counts.add(j)
        if int(row["workload_mix_size"]) != j or len(weights) != j or not np.isclose(sum(weights), 1.0, atol=1e-8):
            bad_weights.append(row_id)
        if len(qos) != j or any((value < 0 or value > 1 or not np.isfinite(value)) for value in qos):
            bad_qos.append(row_id)
    if bad_mix or bad_weights or bad_qos:
        raise AuditError(
            f"Vector validation failed: bad_mix={bad_mix[:10]}, bad_weights={bad_weights[:10]}, bad_qos={bad_qos[:10]}"
        )
    return {"job_counts": sorted(job_counts)}


def add_training_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["weights"] = frame["weights"].map(lambda value: json.dumps(parse_vector(value)))
    frame["QoS_Delay_Probabilities"] = frame["QoS_Delay_Probabilities"].map(
        lambda value: json.dumps(parse_vector(value))
    )
    frame["actual_max_pj"] = frame["QoS_Delay_Probabilities"].map(lambda value: max(parse_vector(value)))
    frame["actual_mean_pj"] = frame["QoS_Delay_Probabilities"].map(lambda value: float(np.mean(parse_vector(value))))
    frame["Actual_Tracking_Pass"] = pd.to_numeric(frame["Ctrack_Epsilon_90th"], errors="raise") <= TRACKING_THRESHOLD
    frame["Actual_QoS_Pass"] = frame["actual_max_pj"] <= QOS_THRESHOLD
    frame["Actual_Both_Pass"] = frame["Actual_Tracking_Pass"] & frame["Actual_QoS_Pass"]
    frame["Tracking_Decision_Band"] = pd.to_numeric(frame["Ctrack_Epsilon_90th"], errors="raise").between(0.20, 0.40)
    frame["QoS_Decision_Band"] = frame["actual_max_pj"].between(0.05, 0.15)

    if "Base_Plan_Row_ID" not in frame.columns:
        frame["Base_Plan_Row_ID"] = frame["Plan_Row_ID"].astype(str)
    frame["Base_Plan_Row_ID"] = frame["Base_Plan_Row_ID"].astype(str)
    group_sizes = frame.groupby("Base_Plan_Row_ID")["Plan_Row_ID"].transform("size").astype(int)
    frame["Base_Group_Row_Count"] = group_sizes
    frame["Base_Group_Normalization_Weight"] = 1.0 / group_sizes.astype(float)

    if {
        "Simulator_RSR_Total_Cost",
        "Ctrack_Weighted_Cost",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "Diagnostic_FullPaperObjective_Cost",
    }.issubset(frame.columns):
        reconstructed = (
            pd.to_numeric(frame["Simulator_RSR_Total_Cost"], errors="raise")
            + pd.to_numeric(frame["Ctrack_Weighted_Cost"], errors="raise")
            + pd.to_numeric(frame["Diagnostic_FlexDC_SoftPlus_QoS_Cost"], errors="raise")
        )
        reported = pd.to_numeric(frame["Diagnostic_FullPaperObjective_Cost"], errors="raise")
        frame["Recomputed_FullPaperObjective_Cost"] = reconstructed
        frame["FullObjective_Reconstruction_Error"] = reconstructed - reported
        max_error = float(np.max(np.abs(reconstructed - reported)))
        if max_error > 1e-7:
            raise AuditError(f"Full objective reconstruction max error={max_error}")
    return frame


def validate_training_schema(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [column for column in TRAINING_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise AuditError(
            "Training-ready schema is incomplete. Missing columns: " + ", ".join(missing)
        )
    vector_audit = validate_vectors(frame)
    numeric_columns = [
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "Pbar_ratio",
        "R_ratio",
        "server_count",
        "utilization",
        "Mtrack_Error_MeanAbs_Normalized",
        "Ctrack_Epsilon_90th",
        "Simulator_Power_Cost",
        "Simulator_RSR_Total_Cost",
        "Diagnostic_FullPaperObjective_Cost",
        "Mtrack_Price_Coefficient_piE",
        "Mtrack_Hour_Seconds",
        "R_actual_watts",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            raise AuditError(f"Non-finite or nonnumeric values in {column}")
    return vector_audit


def context_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["Workload_Name", "server_count", "utilization"]
    for keys, group in frame.groupby(group_columns, sort=True):
        rows.append({
            "Workload_Name": keys[0],
            "server_count": int(keys[1]),
            "utilization": float(keys[2]),
            "Rows": int(len(group)),
            "Base_Groups": int(group["Base_Plan_Row_ID"].nunique()),
            "Unique_P_R_Points": int(group[find_column(group, "pr_id", "P_R_ID")].nunique()) if find_column(group, "pr_id", "P_R_ID") else None,
            "Unique_Weights": int(group[find_column(group, "weight_id", "Weight_ID")].nunique()) if find_column(group, "weight_id", "Weight_ID") else None,
            "Tracking_Pass": int(group["Actual_Tracking_Pass"].sum()),
            "QoS_Pass": int(group["Actual_QoS_Pass"].sum()),
            "Both_Pass": int(group["Actual_Both_Pass"].sum()),
            "Tracking_Boundary": int(group["Tracking_Decision_Band"].sum()),
            "QoS_Boundary": int(group["QoS_Decision_Band"].sum()),
        })
    return pd.DataFrame(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def parse_model_ids(text: str) -> list[str]:
    if str(text).strip().lower() == "all":
        return list(MODEL_SPECS)
    values = [value.strip() for value in str(text).split(",") if value.strip()]
    return [resolve_model_spec(value).model_id for value in values]



def _plan_path_values(plan: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    for column in candidates:
        if column in plan.columns:
            return sorted({str(value).strip() for value in plan[column].dropna() if str(value).strip()})
    return []


def _resolve_existing_repo_file(value: str | Path, repo_root: Path) -> Path | None:
    candidate = Path(value).expanduser()
    options = [candidate] if candidate.is_absolute() else [repo_root / candidate]
    # Worker outputs sometimes store paths relative to src/peacsim.
    if not candidate.is_absolute():
        options += [repo_root / "src" / "peacsim" / candidate]
    for option in options:
        try:
            resolved = option.resolve()
        except Exception:
            continue
        if resolved.exists() and resolved.is_file():
            try:
                resolved.relative_to(repo_root)
            except ValueError:
                continue
            return resolved
    return None


def _discover_ini_references(config_path: Path, repo_root: Path) -> list[Path]:
    """Collect repository-local files referenced by an INI config."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config_path, encoding="utf-8")
    discovered: list[Path] = []
    for section in parser.sections():
        for _, raw_value in parser.items(section):
            text = str(raw_value).strip().strip('"').strip("'")
            if not text or text.lower() in {"true", "false", "none"}:
                continue
            # Ignore obvious scalar/list values.
            if re.fullmatch(r"[-+0-9.eE, ]+", text):
                continue
            resolved = _resolve_existing_repo_file(text, repo_root)
            if resolved is None:
                # Config-relative references are also common.
                local = (config_path.parent / text).resolve()
                if local.exists() and local.is_file():
                    try:
                        local.relative_to(repo_root)
                    except ValueError:
                        pass
                    else:
                        resolved = local
            if resolved is not None:
                discovered.append(resolved)
    return discovered


def collect_runtime_bundle_files(
    *,
    plan: pd.DataFrame,
    plan_path: Path,
    repo_root: Path,
    gradient_config: str | None,
    cluster_config: str | None,
    extra_runtime_files: Sequence[str],
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Collect custom files required for Colab inference/simulator validation.

    This closes the gap in the previous generic package: generated workload and
    gradient configs are not necessarily in a fresh public FlexDC clone, so the
    dataset bundle carries exact hashed copies of them.
    """
    values: list[str] = []
    values.extend(_plan_path_values(plan, ["workload_config", "Workload_Config", "job_config", "Job_Config"]))
    values.extend(_plan_path_values(plan, ["experiment_config", "Experiment_Config"]))
    if gradient_config:
        values.append(gradient_config)
    if cluster_config:
        values.append(cluster_config)
    values.extend(extra_runtime_files)

    # Preserve exact planning/execution scripts when present.
    standard_candidates = [
        "src/peacsim/am_data_extraction_wizard.py",
        "am_run_flexdc_dataset_parallel_combo.py",
        "am_generate_flexdc_j2_pair_workloads.py",
        "am_generate_flexdc_exact_sweep_plan.py",
        "J2_PAIRWISE_SWEEP_COMMANDS.md",
        "requirements.txt",
        "pyproject.toml",
    ]
    values.extend(standard_candidates)

    resolved: dict[Path, None] = {}
    resolved[plan_path.resolve()] = None
    for value in values:
        path = _resolve_existing_repo_file(value, repo_root)
        if path is not None:
            resolved[path] = None

    # Recursively include repository-local input data referenced by the copied
    # experiment/gradient/cluster INIs (e.g. ISO signal CSVs).
    for path in list(resolved):
        if path.suffix.lower() == ".ini":
            for reference in _discover_ini_references(path, repo_root):
                resolved[reference] = None

    files = sorted(resolved, key=lambda path: path.relative_to(repo_root).as_posix())
    manifest: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(repo_root).as_posix()
        manifest.append({
            "relative_path": relative,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return files, manifest


def split_file_for_upload(path: Path, *, part_size_mib: int, manifest_path: Path) -> tuple[list[Path], dict[str, Any]]:
    """Split a large archive into browser-upload-safe parts."""
    if part_size_mib <= 0:
        payload = {"status": "SKIPPED", "archive": str(path), "parts": []}
        write_json(manifest_path, payload)
        return [], payload
    part_size = int(part_size_mib) * 1024 * 1024
    parts: list[Path] = []
    with path.open("rb") as source:
        index = 1
        while True:
            block = source.read(part_size)
            if not block:
                break
            part = path.with_name(f"{path.name}.part{index:03d}")
            part.write_bytes(block)
            parts.append(part)
            index += 1
    payload = {
        "status": "PASS",
        "archive_name": path.name,
        "archive_size_bytes": int(path.stat().st_size),
        "archive_sha256": sha256_file(path),
        "part_size_mib": int(part_size_mib),
        "parts": [
            {
                "name": part.name,
                "size_bytes": int(part.stat().st_size),
                "sha256": sha256_file(part),
            }
            for part in parts
        ],
    }
    write_json(manifest_path, payload)
    return parts, payload

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True, help="Immutable exact sweep plan CSV.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--launch-manifest")
    source.add_argument("--optimization-dir")
    source.add_argument("--results-csv")
    parser.add_argument("--diagnostics-csv")
    parser.add_argument("--prefix", help="Worker output directory prefix with --optimization-dir.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="flexdc_training_dataset")
    parser.add_argument("--output-prefix", default="flexdc")
    parser.add_argument("--models", default="all", help="all or comma-separated model IDs.")
    parser.add_argument("--profile-split-seed", type=int, default=20260804)
    parser.add_argument("--allow-non-j2", action="store_true", help="Do not require exactly two jobs per workload.")
    parser.add_argument(
        "--include-runtime-bundle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Package generated FlexDC configs/scripts needed by Colab inference.",
    )
    parser.add_argument(
        "--gradient-config",
        default="configs/gradient_descent/gradient_descent_j2_pairwise_rsr.ini",
        help="Generated gradient config to preserve in the runtime bundle.",
    )
    parser.add_argument(
        "--cluster-config",
        default="configs/cluster/cluster.ini",
        help="Cluster config to preserve in the runtime bundle.",
    )
    parser.add_argument(
        "--runtime-file",
        action="append",
        default=[],
        help="Additional repository-relative runtime file; may be repeated.",
    )
    parser.add_argument(
        "--colab-part-size-mib",
        type=int,
        default=20,
        help="Also split the final ZIP into upload-safe parts; 0 disables.",
    )
    parser.add_argument("--no-zip", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    plan_path = resolve_repo_path(args.plan_file, repo_root)
    out_dir = resolve_repo_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.output_prefix)

    plan = pd.read_csv(plan_path, low_memory=False)
    plan = collapse_weight_columns(plan)
    plan = canonicalize_common_columns(plan)
    if "Plan_Row_ID" not in plan.columns:
        raise AuditError("Plan lacks plan_row_id/Plan_Row_ID.")
    plan["Plan_Row_ID"] = plan["Plan_Row_ID"].astype(str)
    if plan["Plan_Row_ID"].duplicated().any():
        raise AuditError("Plan contains duplicate Plan_Row_ID values.")

    if args.launch_manifest:
        results_paths, diagnostics_paths = collect_from_manifest(resolve_repo_path(args.launch_manifest, repo_root))
    elif args.optimization_dir:
        if not args.prefix:
            raise ValueError("--prefix is required with --optimization-dir")
        results_paths, diagnostics_paths = collect_from_scan(resolve_repo_path(args.optimization_dir, repo_root), args.prefix)
    else:
        if not args.diagnostics_csv:
            raise ValueError("--diagnostics-csv is required with --results-csv")
        results_paths = [Path(args.results_csv)]
        diagnostics_paths = [Path(args.diagnostics_csv)]
    results_paths = resolve_paths(results_paths, repo_root)
    diagnostics_paths = resolve_paths(diagnostics_paths, repo_root)

    results = read_and_combine(results_paths, "results")
    diagnostics = read_and_combine(diagnostics_paths, "diagnostics")
    merged = merge_outputs_with_plan(plan, results, diagnostics)
    merged = canonicalize_common_columns(merged)
    merged = add_config_derived_metadata(merged, repo_root)
    merged = canonicalize_common_columns(merged)
    merged = add_training_labels(merged)
    schema_audit = validate_training_schema(merged)

    expected_jobs = None if args.allow_non_j2 else 2
    if expected_jobs is not None and schema_audit["job_counts"] != [expected_jobs]:
        raise AuditError(f"Expected only J={expected_jobs}; found {schema_audit['job_counts']}")

    combined_results_path = out_dir / f"{output_prefix}_combined_grid_search_results.csv"
    combined_diagnostics_path = out_dir / f"{output_prefix}_combined_grid_search_diagnostics.csv"
    master_path = out_dir / f"{output_prefix}_master_training_ready.csv"
    context_path = out_dir / f"{output_prefix}_master_context_summary.csv"
    results.to_csv(combined_results_path, index=False)
    diagnostics.to_csv(combined_diagnostics_path, index=False)
    merged.to_csv(master_path, index=False)
    master_context = context_summary(merged)
    master_context.to_csv(context_path, index=False)

    model_ids = parse_model_ids(args.models)
    manifest_rows: list[dict[str, Any]] = []
    all_outputs: list[Path] = [
        combined_results_path,
        combined_diagnostics_path,
        master_path,
        context_path,
    ]
    model_audits: dict[str, Any] = {}
    for model_id in model_ids:
        split_frame, audit = assign_profile_holdout_split(
            merged,
            model_id,
            split_seed=args.profile_split_seed,
            expected_jobs=expected_jobs,
            strict_master_coverage=(expected_jobs == 2),
        )
        model_path = out_dir / f"{output_prefix}_{model_id}_training_ready.csv"
        row_manifest_path = out_dir / f"{output_prefix}_{model_id}_row_manifest.csv"
        split_summary_path = out_dir / f"{output_prefix}_{model_id}_split_summary.csv"
        workload_summary_path = out_dir / f"{output_prefix}_{model_id}_workload_summary.csv"
        audit_path = out_dir / f"{output_prefix}_{model_id}_audit.json"

        split_frame.to_csv(model_path, index=False)
        manifest_columns = [
            "Plan_Row_ID", "Base_Plan_Row_ID", "Workload_Name", "Context_ID",
            "server_count", "utilization", "Data_Split", "Benchmark_Type",
            "Workload_Type", "Job_Profiles", "Heldout_Profile_Count",
            "Heldout_Profiles", "Model_ID", "Model_Display_Name", "Split_Type",
        ]
        split_frame[[column for column in manifest_columns if column in split_frame.columns]].to_csv(
            row_manifest_path, index=False
        )
        pd.DataFrame(audit["split_summary"]).to_csv(split_summary_path, index=False)
        pd.DataFrame(audit["workload_summary"]).to_csv(workload_summary_path, index=False)
        audit.update({
            "plan_file": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "master_training_ready": str(master_path),
            "training_ready": str(model_path),
            "row_manifest": str(row_manifest_path),
            "split_summary_csv": str(split_summary_path),
            "workload_summary_csv": str(workload_summary_path),
        })
        write_json(audit_path, audit)
        model_audits[model_id] = audit
        all_outputs.extend([
            model_path,
            row_manifest_path,
            split_summary_path,
            workload_summary_path,
            audit_path,
        ])
        manifest_rows.append({
            "Model_ID": model_id,
            "Model_Display_Name": audit["model_spec"]["display_name"],
            "Split_Type": audit["model_spec"]["split_type"],
            "Training_Ready_CSV": str(model_path),
            "Rows": audit["rows"],
            "Base_Groups": audit["base_groups"],
            "Seen_Profile_Workloads": audit["workload_counts"]["seen_profile_workloads"],
            "One_Heldout_Workloads": audit["workload_counts"]["one_heldout_workloads"],
            "Two_Heldout_Workloads": audit["workload_counts"]["two_heldout_workloads"],
            "Status": audit["status"],
        })

    dataset_manifest = pd.DataFrame(manifest_rows)
    dataset_manifest_path = out_dir / f"{output_prefix}_dataset_manifest.csv"
    dataset_manifest.to_csv(dataset_manifest_path, index=False)
    all_outputs.append(dataset_manifest_path)

    runtime_files: list[Path] = []
    runtime_manifest_rows: list[dict[str, Any]] = []
    runtime_manifest_path = out_dir / f"{output_prefix}_runtime_manifest.json"
    if args.include_runtime_bundle:
        runtime_files, runtime_manifest_rows = collect_runtime_bundle_files(
            plan=plan,
            plan_path=plan_path,
            repo_root=repo_root,
            gradient_config=args.gradient_config,
            cluster_config=args.cluster_config,
            extra_runtime_files=args.runtime_file,
        )
        runtime_payload = {
            "status": "PASS",
            "repo_root_at_packaging": str(repo_root),
            "file_count": len(runtime_manifest_rows),
            "files": runtime_manifest_rows,
        }
        write_json(runtime_manifest_path, runtime_payload)
        all_outputs.append(runtime_manifest_path)
    else:
        write_json(runtime_manifest_path, {"status": "SKIPPED", "file_count": 0, "files": []})
        all_outputs.append(runtime_manifest_path)

    master_audit = {
        "status": "PASS",
        "repo_root": str(repo_root),
        "plan_file": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "plan_rows": int(len(plan)),
        "result_rows": int(len(results)),
        "diagnostic_rows": int(len(diagnostics)),
        "master_training_ready_rows": int(len(merged)),
        "contexts": int(len(master_context)),
        "workloads": int(merged["Workload_Name"].nunique()),
        "job_counts": schema_audit["job_counts"],
        "actual_tracking_pass_rows": int(merged["Actual_Tracking_Pass"].sum()),
        "actual_qos_pass_rows": int(merged["Actual_QoS_Pass"].sum()),
        "actual_both_pass_rows": int(merged["Actual_Both_Pass"].sum()),
        "source_results_files": [str(path) for path in results_paths],
        "source_diagnostics_files": [str(path) for path in diagnostics_paths],
        "model_datasets": model_audits,
        "runtime_bundle_enabled": bool(args.include_runtime_bundle),
        "runtime_bundle_file_count": int(len(runtime_manifest_rows)),
        "runtime_manifest": str(runtime_manifest_path),
        "outputs": [str(path) for path in all_outputs],
    }
    master_audit_path = out_dir / f"{output_prefix}_master_audit.json"
    write_json(master_audit_path, master_audit)
    all_outputs.append(master_audit_path)

    zip_path = None
    if not args.no_zip:
        zip_path = out_dir / f"{output_prefix}_profile_holdout_training_bundle.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in all_outputs:
                archive.write(path, arcname=f"dataset/{path.name}")
            for path in runtime_files:
                relative = path.relative_to(repo_root).as_posix()
                archive.write(path, arcname=f"runtime_bundle/FlexDC/{relative}")

    upload_parts: list[Path] = []
    upload_manifest_path = out_dir / f"{output_prefix}_profile_holdout_training_bundle_upload_manifest.json"
    if zip_path is not None:
        upload_parts, _ = split_file_for_upload(
            zip_path,
            part_size_mib=args.colab_part_size_mib,
            manifest_path=upload_manifest_path,
        )

    print("FlexDC generic dataset preparation")
    print("----------------------------------")
    print("Status: PASS")
    print(f"Plan rows: {len(plan):,}")
    print(f"Worker result files: {len(results_paths)}")
    print(f"Worker diagnostic files: {len(diagnostics_paths)}")
    print(f"Master training-ready rows: {len(merged):,}")
    print(f"Workloads: {merged['Workload_Name'].nunique()}")
    print(f"Contexts: {len(master_context)}")
    print(f"Job counts: {schema_audit['job_counts']}")
    print(f"Output directory: {out_dir}")
    print()
    print(dataset_manifest.to_string(index=False))
    print(f"Runtime bundle files: {len(runtime_files)}")
    if zip_path is not None:
        print(f"\nColab dataset + runtime ZIP: {zip_path}")
        if upload_parts:
            print(f"Upload-safe parts ({len(upload_parts)}):")
            for part in upload_parts:
                print(" -", part)
            print("Upload manifest:", upload_manifest_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nFAILED:", file=sys.stderr)
        print(exc, file=sys.stderr)
        raise
