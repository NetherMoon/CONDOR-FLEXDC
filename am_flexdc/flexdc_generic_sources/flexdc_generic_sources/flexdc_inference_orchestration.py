"""Reusable orchestration for generic CONDOR–FlexDC inference notebooks.

This module restores the operational capabilities that lived directly in the
original V3 and paired-comparison notebooks while keeping model names generic:

* predict one point;
* deterministic shared-start multi-model optimization;
* trajectory and top-k persistence;
* real FlexDC validation of the anchor and top-k candidates across seeds;
* resumable fixed suites, custom rounds, and all-workload batches;
* identical-point comparisons and external-reference comparisons;
* completion manifests that make interrupted notebook runs recoverable.

The numerical model, feature construction, objective reconstruction, and
simulator parsing remain in ``flexdc_behavior_inference_utilities``.  This file
only coordinates those tested primitives.
"""
from __future__ import annotations

import configparser
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from flexdc_behavior_inference_utilities import (
    ExperimentSpec,
    LoadedBehaviorModel,
    OptimizationSettings,
    calculate_pr_bounds,
    dataframe_for_csv,
    optimize_candidates,
    predict_configuration,
    read_experiment_config,
    read_workload_config,
    resolve_safety_limits,
    run_flexdc_validation,
)
from flexdc_profile_split import (
    MODEL_SPECS,
    parse_profiles_from_text,
    resolve_model_spec,
    workload_type,
)


def safe_tag(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "run"


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if pd.isna(value) if not isinstance(value, (str, bytes, bool)) else False:
        return None
    return value


def _weights(value: Any) -> list[float]:
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            pass
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    raise TypeError(f"Cannot parse weights from {type(value).__name__}: {value!r}")


@dataclass(frozen=True)
class ScenarioDefinition:
    case_id: str
    workload_config: str
    experiment_config: str
    server_count: int | None = None
    utilization: float | None = None
    simulation_duration: int | None = None
    initial_pbar: float | None = None
    initial_r: float | None = None
    initial_weights: tuple[float, ...] | None = None
    category: str | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScenarioDefinition":
        aliases = {
            "workload": "workload_config",
            "experiment": "experiment_config",
            "pbar": "initial_pbar",
            "r": "initial_r",
            "weights": "initial_weights",
            "N": "server_count",
            "U": "utilization",
            "duration": "simulation_duration",
        }
        payload = dict(value)
        for old, new in aliases.items():
            if old in payload and new not in payload:
                payload[new] = payload.pop(old)
        if payload.get("initial_weights") is not None:
            payload["initial_weights"] = tuple(_weights(payload["initial_weights"]))
        if "case_id" not in payload:
            payload["case_id"] = safe_tag(Path(str(payload["workload_config"])).stem)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class ScenarioRunResult:
    case: ScenarioDefinition
    model_label: str
    output_dir: str
    prediction_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    prediction_per_job: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_starts: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_k: pd.DataFrame = field(default_factory=pd.DataFrame)
    trajectory: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_per_job: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_aggregate: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)


def build_workload_catalog(workload_dir: str | Path) -> dict[str, Path]:
    workload_dir = Path(workload_dir).expanduser().resolve()
    if not workload_dir.exists():
        raise FileNotFoundError(workload_dir)
    catalog = {path.stem: path for path in sorted(workload_dir.glob("*.ini"))}
    if not catalog:
        raise FileNotFoundError(f"No workload INIs found under {workload_dir}")
    return catalog


def classify_workload_for_model(workload_name: str, model_id: str | None) -> dict[str, Any]:
    profiles = parse_profiles_from_text(workload_name)
    result = {
        "Workload_Name": Path(str(workload_name)).stem,
        "Profiles": profiles,
        "Workload_Type": workload_type(profiles) if profiles else "Unknown",
        "Heldout_Profile_Count": np.nan,
        "Benchmark_Type": "Unclassified",
    }
    if not model_id or model_id not in MODEL_SPECS or not profiles:
        return result
    spec = resolve_model_spec(model_id)
    held = sum(profile in spec.heldout_profiles for profile in profiles)
    result["Heldout_Profile_Count"] = int(held)
    result["Benchmark_Type"] = (
        "Seen-Profile Baseline"
        if held == 0
        else "One Held-Out Profile"
        if held == 1
        else "Two Held-Out Profiles"
    )
    return result


def build_profile_suite(
    workload_catalog: Mapping[str, Path],
    *,
    model_id: str | None,
    per_benchmark: int = 1,
    preferred_workload_types: Sequence[str] = ("Mixed", "Inference-only", "Training-only"),
) -> pd.DataFrame:
    records = []
    for name, path in workload_catalog.items():
        record = classify_workload_for_model(name, model_id)
        record["Path"] = str(path)
        records.append(record)
    table = pd.DataFrame(records)
    if len(table) == 0:
        return table
    selected: list[pd.DataFrame] = []
    benchmark_order = ["Seen-Profile Baseline", "One Held-Out Profile", "Two Held-Out Profiles"]
    for benchmark in benchmark_order:
        group = table[table["Benchmark_Type"] == benchmark].copy()
        if len(group) == 0:
            continue
        group["_type_order"] = group["Workload_Type"].map(
            {name: index for index, name in enumerate(preferred_workload_types)}
        ).fillna(len(preferred_workload_types))
        group = group.sort_values(["_type_order", "Workload_Name"])
        selected.append(group.head(int(per_benchmark)).drop(columns=["_type_order"]))
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=table.columns)


def experiment_for_seed(
    base_path: str | Path,
    seed: int,
    target_dir: str | Path,
    *,
    utilization: float | None = None,
    server_count: int | None = None,
    simulation_duration: int | None = None,
) -> Path:
    base_path = Path(base_path).expanduser().resolve()
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(base_path):
        raise FileNotFoundError(base_path)
    if not parser.has_section("system"):
        raise KeyError(f"Missing [system] in {base_path}")
    parser.set("system", "random_seed", str(int(seed)))
    if utilization is not None:
        parser.set("system", "utilization", str(float(utilization)))
    if server_count is not None:
        parser.set("system", "server_count", str(int(server_count)))
    if simulation_duration is not None:
        parser.set("system", "simulation_duration", str(int(simulation_duration)))
    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{base_path.stem}_seed_{int(seed)}.ini"
    with target.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return target


def _candidate_rows(
    *,
    prediction_summary: pd.DataFrame,
    top_k: pd.DataFrame,
    validate_anchor: bool,
    validate_top_k: bool,
    initial_pbar: float,
    initial_r: float,
    initial_weights: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if validate_anchor:
        prediction = prediction_summary.iloc[0].to_dict() if len(prediction_summary) else {}
        rows.append(
            {
                "Candidate_Type": "Anchor",
                "Candidate_Rank": 0,
                "Pbar_kw_per_server": float(initial_pbar),
                "R_kw_per_server": float(initial_r),
                "weights": list(map(float, initial_weights)),
                **prediction,
            }
        )
    if validate_top_k:
        for index, row in top_k.iterrows():
            record = row.to_dict()
            record["Candidate_Type"] = "Optimized"
            record["Candidate_Rank"] = int(record.get("Candidate_Rank", index + 1))
            record["weights"] = _weights(record["weights"])
            rows.append(record)
    return rows


def aggregate_validation(validation: pd.DataFrame) -> pd.DataFrame:
    if validation is None or len(validation) == 0:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ["Case_ID", "Model", "Candidate_Type", "Candidate_Rank"]
        if column in validation.columns
    ]
    aggregation: dict[str, tuple[str, str]] = {
        "Runs": ("Seed", "size"),
        "Seeds_Passed": ("Actual_Both_Pass", "sum"),
        "Worst_P90": ("Actual_P90_Tracking", "max"),
        "Worst_Max_Pj": ("Actual_Max_Pj", "max"),
        "Mean_Actual_Objective": ("Actual_Full_Objective", "mean"),
    }
    aggregation = {
        name: spec
        for name, spec in aggregation.items()
        if spec[0] in validation.columns
    }
    result = validation.groupby(group_columns, dropna=False).agg(**aggregation).reset_index()
    if "Runs" in result and "Seeds_Passed" in result:
        result["All_Seeds_Pass"] = result["Seeds_Passed"] == result["Runs"]
    return result


def run_scenario_case(
    *,
    loaded: LoadedBehaviorModel,
    model_label: str,
    case: ScenarioDefinition | Mapping[str, Any],
    output_dir: str | Path,
    settings: OptimizationSettings,
    tracking_margin: float = 0.04,
    qos_margin: float = 0.01,
    run_predict: bool = True,
    run_optimize: bool = True,
    run_flexdc: bool = False,
    validate_anchor: bool = True,
    validate_top_k: bool = True,
    simulator_seeds: Sequence[int] = (30, 31, 32),
    flexdc_root: str | Path | None = None,
    gradient_config: str | Path | None = None,
    cluster_config: str | Path | None = None,
    policy_name: str = "AQA",
    node_count_control: bool = True,
    validation_timeout_seconds: int = 1800,
    python_executable: str = sys.executable,
    dry_run_flexdc: bool = False,
    resume: bool = True,
) -> ScenarioRunResult:
    """Execute one complete Predict → Optimize → FlexDC workflow."""
    if not isinstance(case, ScenarioDefinition):
        case = ScenarioDefinition.from_mapping(case)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "case_summary.json"

    # A complete, non-dry-run case can be reused after interruption.
    if resume and completion_path.exists():
        summary = json.loads(completion_path.read_text(encoding="utf-8"))
        prediction = pd.read_csv(output_dir / "predict_one_summary.csv") if (output_dir / "predict_one_summary.csv").exists() else pd.DataFrame()
        per_job = pd.read_csv(output_dir / "predict_one_per_job.csv") if (output_dir / "predict_one_per_job.csv").exists() else pd.DataFrame()
        starts = pd.read_csv(output_dir / "all_starts.csv") if (output_dir / "all_starts.csv").exists() else pd.DataFrame()
        top = pd.read_csv(output_dir / "top_k.csv") if (output_dir / "top_k.csv").exists() else pd.DataFrame()
        trajectory = pd.read_csv(output_dir / "trajectory.csv") if (output_dir / "trajectory.csv").exists() else pd.DataFrame()
        validation = pd.read_csv(output_dir / "predicted_vs_actual.csv") if (output_dir / "predicted_vs_actual.csv").exists() else pd.DataFrame()
        validation_jobs = pd.read_csv(output_dir / "validation_per_job.csv") if (output_dir / "validation_per_job.csv").exists() else pd.DataFrame()
        aggregate = pd.read_csv(output_dir / "validation_aggregate.csv") if (output_dir / "validation_aggregate.csv").exists() else pd.DataFrame()
        return ScenarioRunResult(
            case=case,
            model_label=model_label,
            output_dir=str(output_dir),
            prediction_summary=prediction,
            prediction_per_job=per_job,
            all_starts=starts,
            top_k=top,
            trajectory=trajectory,
            validation_summary=validation,
            validation_per_job=validation_jobs,
            validation_aggregate=aggregate,
            summary=summary,
        )

    workload_path = Path(case.workload_config).expanduser().resolve()
    experiment_path = Path(case.experiment_config).expanduser().resolve()
    if case.simulation_duration is not None:
        experiment_path = experiment_for_seed(
            experiment_path,
            seed=read_experiment_config(experiment_path).random_seed,
            target_dir=output_dir / "case_experiment",
            utilization=case.utilization,
            server_count=case.server_count,
            simulation_duration=case.simulation_duration,
        )
    workload = read_workload_config(workload_path)
    experiment = read_experiment_config(
        experiment_path,
        server_count_override=case.server_count,
        utilization_override=case.utilization,
    )
    bounds = calculate_pr_bounds(
        workload,
        pbar_lower_factor=0.9,
        pbar_upper_factor=1.0,
        pr_upper_factor=1.2,
        r_lower_kw_per_server=0.01,
    )
    safety = resolve_safety_limits(
        loaded.constants,
        tracking_margin=tracking_margin,
        qos_margin=qos_margin,
    )

    initial_pbar = float(
        case.initial_pbar
        if case.initial_pbar is not None
        else 0.5 * (bounds.pbar_lower_kw_per_server + bounds.pbar_upper_kw_per_server)
    )
    initial_r = float(
        case.initial_r
        if case.initial_r is not None
        else min(0.3 * initial_pbar, 0.5 * bounds.r_lower_kw_per_server + 0.25 * initial_pbar)
    )
    initial_weights = (
        list(case.initial_weights)
        if case.initial_weights is not None
        else [1.0 / workload.job_count] * workload.job_count
    )

    prediction_summary = pd.DataFrame()
    prediction_per_job = pd.DataFrame()
    if run_predict:
        summary, jobs = predict_configuration(
            loaded,
            workload=workload,
            experiment=experiment,
            pbar_kw_per_server=initial_pbar,
            r_kw_per_server=initial_r,
            weights=initial_weights,
            safety=safety,
        )
        prediction_summary = pd.DataFrame(
            [{"Case_ID": case.case_id, "Model": model_label, **summary}]
        )
        prediction_per_job = jobs.copy()
        if len(prediction_per_job):
            prediction_per_job.insert(0, "Case_ID", case.case_id)
            prediction_per_job.insert(1, "Model", model_label)
        dataframe_for_csv(prediction_summary).to_csv(output_dir / "predict_one_summary.csv", index=False)
        dataframe_for_csv(prediction_per_job).to_csv(output_dir / "predict_one_per_job.csv", index=False)

    all_starts = pd.DataFrame()
    top_k = pd.DataFrame()
    trajectory = pd.DataFrame()
    if run_optimize:
        all_starts, top_k, trajectory = optimize_candidates(
            loaded,
            workload=workload,
            experiment=experiment,
            bounds=bounds,
            safety=safety,
            settings=settings,
            initial_pbar=initial_pbar,
            initial_reserve=initial_r,
            initial_weights=initial_weights,
        )
        for frame in [all_starts, top_k, trajectory]:
            if len(frame):
                frame.insert(0, "Case_ID", case.case_id)
                frame.insert(1, "Model", model_label)
        dataframe_for_csv(all_starts).to_csv(output_dir / "all_starts.csv", index=False)
        dataframe_for_csv(top_k).to_csv(output_dir / "top_k.csv", index=False)
        dataframe_for_csv(trajectory).to_csv(output_dir / "trajectory.csv", index=False)

    validation_rows: list[dict[str, Any]] = []
    validation_jobs: list[pd.DataFrame] = []
    if run_flexdc:
        required = {
            "flexdc_root": flexdc_root,
            "gradient_config": gradient_config,
            "cluster_config": cluster_config,
        }
        missing = [name for name, value in required.items() if value is None or not Path(value).exists()]
        if missing:
            raise FileNotFoundError(
                "Real FlexDC validation requires existing: " + ", ".join(missing)
            )
        candidates = _candidate_rows(
            prediction_summary=prediction_summary,
            top_k=top_k,
            validate_anchor=validate_anchor,
            validate_top_k=validate_top_k,
            initial_pbar=initial_pbar,
            initial_r=initial_r,
            initial_weights=initial_weights,
        )
        seed_dir = output_dir / "seed_experiments"
        for candidate in candidates:
            for seed in simulator_seeds:
                seeded_experiment = experiment_for_seed(
                    experiment_path,
                    int(seed),
                    seed_dir,
                    utilization=experiment.utilization,
                    server_count=experiment.server_count,
                    simulation_duration=case.simulation_duration,
                )
                label = safe_tag(
                    f"{model_label}_{case.case_id}_{candidate['Candidate_Type']}_"
                    f"r{candidate['Candidate_Rank']}_s{seed}"
                )
                actual, jobs = run_flexdc_validation(
                    python_executable=python_executable,
                    flexdc_root=flexdc_root,
                    gradient_config=gradient_config,
                    experiment_config=seeded_experiment,
                    cluster_config=cluster_config,
                    workload_config=workload_path,
                    output_label=label,
                    pbar_kw_per_server=float(candidate["Pbar_kw_per_server"]),
                    r_kw_per_server=float(candidate["R_kw_per_server"]),
                    weights=_weights(candidate["weights"]),
                    utilization=experiment.utilization,
                    constants=loaded.constants,
                    policy_name=policy_name,
                    node_count_control=node_count_control,
                    timeout_seconds=validation_timeout_seconds,
                    dry_run=dry_run_flexdc,
                )
                validation_rows.append(
                    {
                        "Case_ID": case.case_id,
                        "Model": model_label,
                        "Seed": int(seed),
                        **candidate,
                        **actual,
                    }
                )
                if len(jobs):
                    jobs = jobs.copy()
                    jobs.insert(0, "Case_ID", case.case_id)
                    jobs.insert(1, "Model", model_label)
                    jobs.insert(2, "Seed", int(seed))
                    jobs.insert(3, "Candidate_Type", candidate["Candidate_Type"])
                    jobs.insert(4, "Candidate_Rank", candidate["Candidate_Rank"])
                    validation_jobs.append(jobs)

    validation_summary = pd.DataFrame(validation_rows)
    validation_per_job = (
        pd.concat(validation_jobs, ignore_index=True) if validation_jobs else pd.DataFrame()
    )
    validation_aggregate = aggregate_validation(validation_summary)
    dataframe_for_csv(validation_summary).to_csv(output_dir / "predicted_vs_actual.csv", index=False)
    dataframe_for_csv(validation_per_job).to_csv(output_dir / "validation_per_job.csv", index=False)
    dataframe_for_csv(validation_aggregate).to_csv(output_dir / "validation_aggregate.csv", index=False)

    selected_actual = None
    if len(validation_summary) and "Actual_Both_Pass" in validation_summary:
        feasible = validation_summary[validation_summary["Actual_Both_Pass"].astype(bool)]
        if len(feasible) and "Actual_Full_Objective" in feasible:
            selected_actual = _jsonable(
                feasible.sort_values("Actual_Full_Objective").iloc[0].to_dict()
            )

    summary = {
        "status": (
            "simulator_feasible_candidate_selected"
            if selected_actual is not None
            else "no_simulator_feasible_candidate"
            if run_flexdc and not dry_run_flexdc
            else "surrogate_optimization_complete"
        ),
        "case": case.to_dict(),
        "model_label": model_label,
        "checkpoint_epoch": int(loaded.checkpoint.get("epoch", -1)),
        "bounds": bounds.to_dict(),
        "safety": safety.to_dict(),
        "settings": _jsonable(settings.__dict__),
        "initial_point": {
            "Pbar_kw_per_server": initial_pbar,
            "R_kw_per_server": initial_r,
            "weights": initial_weights,
        },
        "exact_feasible_starts": int(all_starts["Exact_Both_Pass"].sum())
        if len(all_starts) and "Exact_Both_Pass" in all_starts
        else 0,
        "safety_feasible_starts": int(all_starts["Safety_Both_Pass"].sum())
        if len(all_starts) and "Safety_Both_Pass" in all_starts
        else 0,
        "top_k_count": int(len(top_k)),
        "actual_validations": int(len(validation_summary)),
        "selected_actual_feasible_candidate": selected_actual,
    }
    write_json(completion_path, summary)
    return ScenarioRunResult(
        case=case,
        model_label=model_label,
        output_dir=str(output_dir),
        prediction_summary=prediction_summary,
        prediction_per_job=prediction_per_job,
        all_starts=all_starts,
        top_k=top_k,
        trajectory=trajectory,
        validation_summary=validation_summary,
        validation_per_job=validation_per_job,
        validation_aggregate=validation_aggregate,
        summary=summary,
    )


def run_scenario_suite(
    *,
    model_runtimes: Sequence[Mapping[str, Any]],
    cases: Sequence[ScenarioDefinition | Mapping[str, Any]],
    output_root: str | Path,
    settings: OptimizationSettings,
    tracking_margin: float,
    qos_margin: float,
    run_flexdc: bool,
    validate_anchor: bool,
    validate_top_k: bool,
    simulator_seeds: Sequence[int],
    flexdc_root: str | Path | None,
    gradient_config: str | Path | None,
    cluster_config: str | Path | None,
    validation_timeout_seconds: int,
    resume: bool = True,
    dry_run_flexdc: bool = False,
) -> tuple[list[ScenarioRunResult], pd.DataFrame]:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[ScenarioRunResult] = []
    summaries: list[dict[str, Any]] = []
    for case_value in cases:
        case = case_value if isinstance(case_value, ScenarioDefinition) else ScenarioDefinition.from_mapping(case_value)
        for runtime in model_runtimes:
            label = str(runtime["label"])
            loaded = runtime["loaded"]
            result = run_scenario_case(
                loaded=loaded,
                model_label=label,
                case=case,
                output_dir=output_root / safe_tag(case.case_id) / safe_tag(label),
                settings=settings,
                tracking_margin=tracking_margin,
                qos_margin=qos_margin,
                run_predict=True,
                run_optimize=True,
                run_flexdc=run_flexdc,
                validate_anchor=validate_anchor,
                validate_top_k=validate_top_k,
                simulator_seeds=simulator_seeds,
                flexdc_root=flexdc_root,
                gradient_config=gradient_config,
                cluster_config=cluster_config,
                validation_timeout_seconds=validation_timeout_seconds,
                resume=resume,
                dry_run_flexdc=dry_run_flexdc,
            )
            results.append(result)
            summaries.append(
                {
                    "Case_ID": case.case_id,
                    "Model": label,
                    "Status": result.summary.get("status"),
                    "Safety_Feasible_Starts": result.summary.get("safety_feasible_starts", 0),
                    "Top_K": result.summary.get("top_k_count", 0),
                    "Actual_Validations": result.summary.get("actual_validations", 0),
                    "Actual_Feasible_Selected": result.summary.get("selected_actual_feasible_candidate") is not None,
                }
            )
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output_root / "suite_summary.csv", index=False)
    write_json(output_root / "suite_manifest.json", [result.summary for result in results])
    return results, summary_table


def compare_models_on_points(
    *,
    model_runtimes: Sequence[Mapping[str, Any]],
    workload_config: str | Path,
    experiment_config: str | Path,
    points: Sequence[Mapping[str, Any]],
    server_count: int | None = None,
    utilization: float | None = None,
    tracking_margin: float = 0.04,
    qos_margin: float = 0.01,
) -> pd.DataFrame:
    workload = read_workload_config(workload_config)
    experiment = read_experiment_config(
        experiment_config,
        server_count_override=server_count,
        utilization_override=utilization,
    )
    rows = []
    for point_index, point in enumerate(points):
        point_label = str(point.get("Point", point.get("label", f"Point {point_index + 1}")))
        pbar = float(point.get("Pbar_kw_per_server", point.get("pbar")))
        reserve = float(point.get("R_kw_per_server", point.get("r")))
        weights = _weights(point.get("weights"))
        for runtime in model_runtimes:
            loaded = runtime["loaded"]
            safety = resolve_safety_limits(
                loaded.constants,
                tracking_margin=tracking_margin,
                qos_margin=qos_margin,
            )
            summary, _ = predict_configuration(
                loaded,
                workload=workload,
                experiment=experiment,
                pbar_kw_per_server=pbar,
                r_kw_per_server=reserve,
                weights=weights,
                safety=safety,
            )
            rows.append(
                {
                    "Point": point_label,
                    "Model": runtime["label"],
                    "Pbar_kw_per_server": pbar,
                    "R_kw_per_server": reserve,
                    "weights": weights,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def restore_prior_outputs(prior_root: str | Path | None, output_root: str | Path) -> list[str]:
    """Copy an extracted prior-result tree into the active output tree.

    Existing files are kept unless the prior copy is newer.  This mirrors the
    original paired notebook's interrupted-run recovery without hiding which
    files were restored.
    """
    if prior_root is None:
        return []
    prior_root = Path(prior_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not prior_root.exists():
        raise FileNotFoundError(prior_root)
    restored: list[str] = []
    for source in sorted(path for path in prior_root.rglob("*") if path.is_file()):
        relative = source.relative_to(prior_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or source.stat().st_mtime >= destination.stat().st_mtime:
            destination.write_bytes(source.read_bytes())
            restored.append(relative.as_posix())
    return restored
