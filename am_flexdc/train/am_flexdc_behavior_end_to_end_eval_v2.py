#!/usr/bin/env python3
"""Optimize CONDOR-FlexDC behavior-v2 candidates and validate top-k in FlexDC."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from am_flexdc_behavior_inference_utilities_v2 import (
    OptimizationSettings,
    calculate_pr_bounds,
    dataframe_for_csv,
    load_behavior_model,
    optimize_candidates,
    predict_configuration,
    read_experiment_config,
    read_workload_config,
    resolve_safety_limits,
    run_flexdc_validation,
    score_candidate_table_with_checkpoint,
    write_json,
)


def parse_weights(text: str | None) -> list[float] | None:
    if text is None:
        return None
    return [float(piece.strip()) for piece in text.split(",") if piece.strip()]


def parse_json_list(value) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(x) for x in json.loads(str(value))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--secondary-checkpoint", action="append", default=[])
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--server-count", type=int, default=None)
    parser.add_argument("--utilization", type=float, default=None)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--initial-pbar", type=float, default=None)
    parser.add_argument("--initial-r", type=float, default=None)
    parser.add_argument("--initial-weights", default=None)
    parser.add_argument("--validate-start", action="store_true")

    parser.add_argument("--mode", choices=["pure_objective", "exact_constrained", "margin_constrained"], default="margin_constrained")
    parser.add_argument("--tracking-limit", type=float, default=None)
    parser.add_argument("--qos-limit", type=float, default=None)
    parser.add_argument("--tracking-margin", type=float, default=0.04)
    parser.add_argument("--qos-margin", type=float, default=0.01)

    parser.add_argument("--starts", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-4)
    parser.add_argument("--tracking-penalty", type=float, default=2000.0)
    parser.add_argument("--qos-penalty", type=float, default=2000.0)
    parser.add_argument("--penalty-ramp-fraction", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-distance", type=float, default=0.03)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--near-equal-start-fraction", type=float, default=0.25)
    parser.add_argument("--high-p-low-r-start-fraction", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=25)

    parser.add_argument("--pbar-lower-factor", type=float, default=0.9)
    parser.add_argument("--pbar-upper-factor", type=float, default=1.0)
    parser.add_argument("--pr-upper-factor", type=float, default=1.2)
    parser.add_argument("--r-lower", type=float, default=0.01)
    parser.add_argument("--pbar-min", type=float, default=None)
    parser.add_argument("--pbar-max", type=float, default=None)
    parser.add_argument("--r-min", type=float, default=None)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--r-over-p-max", type=float, default=None)
    parser.add_argument("--weight-min", type=float, default=None)
    parser.add_argument("--weight-max", type=float, default=None)

    parser.add_argument("--flexdc-root", required=True)
    parser.add_argument("--gradient-config", required=True)
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--policy-name", default="AQA")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--validation-timeout", type=int, default=1800)
    parser.add_argument("--dry-run-flexdc", action="store_true")

    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="behavior_v2_e2e")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_behavior_model(args.checkpoint, device_name=args.device)
    workload = read_workload_config(args.workload_config)
    experiment = read_experiment_config(
        args.experiment_config,
        server_count_override=args.server_count,
        utilization_override=args.utilization,
    )
    bounds = calculate_pr_bounds(
        workload,
        pbar_lower_factor=args.pbar_lower_factor,
        pbar_upper_factor=args.pbar_upper_factor,
        pr_upper_factor=args.pr_upper_factor,
        r_lower_kw_per_server=args.r_lower,
    )
    safety = resolve_safety_limits(
        loaded.constants,
        tracking_limit=args.tracking_limit,
        qos_limit=args.qos_limit,
        tracking_margin=args.tracking_margin,
        qos_margin=args.qos_margin,
    )
    settings = OptimizationSettings(
        starts=args.starts,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        mode=args.mode,
        tracking_penalty=args.tracking_penalty,
        qos_penalty=args.qos_penalty,
        penalty_ramp_fraction=args.penalty_ramp_fraction,
        top_k=args.top_k,
        candidate_distance=args.candidate_distance,
        random_seed=args.random_seed,
        near_equal_start_fraction=args.near_equal_start_fraction,
        high_p_low_r_start_fraction=args.high_p_low_r_start_fraction,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        r_over_p_max=args.r_over_p_max,
        pbar_min_override=args.pbar_min,
        pbar_max_override=args.pbar_max,
        r_min_override=args.r_min,
        r_max_override=args.r_max,
        log_every=args.log_every,
    )

    candidates, top_k, trajectory = optimize_candidates(
        loaded,
        workload=workload,
        experiment=experiment,
        bounds=bounds,
        safety=safety,
        settings=settings,
        initial_pbar=args.initial_pbar,
        initial_reserve=args.initial_r,
        initial_weights=parse_weights(args.initial_weights),
    )
    for index, checkpoint in enumerate(args.secondary_checkpoint, start=1):
        top_k = score_candidate_table_with_checkpoint(
            top_k,
            checkpoint_path=checkpoint,
            workload=workload,
            experiment=experiment,
            safety=safety,
            device_name=args.device,
            prefix=f"Secondary_{index}",
        )

    prefix = args.run_name
    dataframe_for_csv(candidates).to_csv(out_dir / f"{prefix}_all_starts.csv", index=False)
    dataframe_for_csv(top_k).to_csv(out_dir / f"{prefix}_top_k_predicted.csv", index=False)
    dataframe_for_csv(trajectory).to_csv(out_dir / f"{prefix}_trajectory.csv", index=False)

    validation_rows: list[dict] = []
    per_job_rows: list[pd.DataFrame] = []

    def validate_candidate(label: str, candidate_rank: int, pbar: float, reserve: float, weights: list[float], predicted: dict) -> None:
        output_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{prefix}_{label}")
        actual, actual_jobs = run_flexdc_validation(
            python_executable=args.python_executable,
            flexdc_root=args.flexdc_root,
            gradient_config=args.gradient_config,
            experiment_config=args.experiment_config,
            cluster_config=args.cluster_config,
            workload_config=args.workload_config,
            output_label=output_label,
            pbar_kw_per_server=pbar,
            r_kw_per_server=reserve,
            weights=weights,
            utilization=experiment.utilization,
            constants=loaded.constants,
            policy_name=args.policy_name,
            node_count_control=True,
            timeout_seconds=args.validation_timeout,
            dry_run=args.dry_run_flexdc,
        )
        row = {
            "Candidate_Label": label,
            "Candidate_Rank": candidate_rank,
            "Pbar_kw_per_server": pbar,
            "R_kw_per_server": reserve,
            "weights": weights,
            **{key: value for key, value in predicted.items() if key.startswith("Predicted_") or key.endswith("Pass") or key.endswith("Slack")},
            **actual,
        }
        if not args.dry_run_flexdc:
            row.update(
                {
                    "P90_Error_PredMinusActual": predicted["Predicted_P90_Tracking"] - actual["Actual_P90_Tracking"],
                    "MaxPj_Error_PredMinusActual": predicted["Predicted_Max_Pj"] - actual["Actual_Max_Pj"],
                    "MRSR_Error_PredMinusActual": predicted["Predicted_M_RSR"] - actual["Actual_M_RSR"],
                    "Objective_Error_PredMinusActual": predicted["Predicted_Full_Objective"] - actual["Actual_Full_Objective"],
                }
            )
            predicted_jobs = np.asarray(predicted["Predicted_QoS_Probabilities"], dtype=float)
            jobs = actual_jobs.copy()
            jobs.insert(0, "Candidate_Label", label)
            jobs.insert(1, "Candidate_Rank", candidate_rank)
            jobs["Job_Type"] = workload.job_names
            jobs["Predicted_Pj"] = predicted_jobs
            jobs["Pj_Error_PredMinusActual"] = predicted_jobs - jobs["Actual_Pj"].to_numpy(float)
            per_job_rows.append(jobs)
        validation_rows.append(row)

    if args.validate_start:
        if args.initial_pbar is None or args.initial_r is None or args.initial_weights is None:
            raise ValueError("--validate-start requires --initial-pbar, --initial-r, and --initial-weights")
        start_weights = parse_weights(args.initial_weights)
        predicted_start, _ = predict_configuration(
            loaded,
            workload=workload,
            experiment=experiment,
            pbar_kw_per_server=args.initial_pbar,
            r_kw_per_server=args.initial_r,
            weights=start_weights,
            safety=safety,
            bounds=bounds,
            r_over_p_max=args.r_over_p_max,
        )
        validate_candidate("start", 0, args.initial_pbar, args.initial_r, start_weights, predicted_start)

    if top_k.empty:
        summary = {
            "status": "no_predicted_feasible_candidate",
            "message": "No top-k candidate met the configured selection thresholds. FlexDC validation was not run for an unconstrained fallback.",
            "safety": safety.to_dict(),
            "settings": settings.__dict__,
            "exact_feasible_starts": int(candidates["Exact_Both_Pass"].sum()),
            "safety_feasible_starts": int(candidates["Safety_Both_Pass"].sum()),
        }
        write_json(out_dir / f"{prefix}_summary.json", summary)
        print(summary["message"])
        return

    for _, row in top_k.iterrows():
        rank = int(row["Candidate_Rank"])
        weights = row["weights"] if isinstance(row["weights"], list) else parse_json_list(row["weights"])
        predicted = row.to_dict()
        validate_candidate(
            f"rank_{rank}",
            rank,
            float(row["Pbar_kw_per_server"]),
            float(row["R_kw_per_server"]),
            weights,
            predicted,
        )

    validation_df = pd.DataFrame(validation_rows)
    dataframe_for_csv(validation_df).to_csv(out_dir / f"{prefix}_predicted_vs_actual.csv", index=False)
    if per_job_rows:
        per_job_df = pd.concat(per_job_rows, ignore_index=True)
        dataframe_for_csv(per_job_df).to_csv(out_dir / f"{prefix}_per_job_qos.csv", index=False)
    else:
        per_job_df = pd.DataFrame()

    if args.dry_run_flexdc:
        final_status = "dry_run_complete"
        selected = None
    else:
        feasible = validation_df[validation_df["Actual_Both_Pass"].astype(bool)].copy()
        if feasible.empty:
            final_status = "no_simulator_feasible_candidate"
            selected = None
        else:
            selected_row = feasible.sort_values("Actual_Full_Objective").iloc[0]
            selected = selected_row.to_dict()
            final_status = "simulator_feasible_candidate_selected"

    summary = {
        "status": final_status,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(loaded.checkpoint.get("epoch", -1)),
        "workload_config": str(Path(args.workload_config).resolve()),
        "experiment_config": str(Path(args.experiment_config).resolve()),
        "safety": safety.to_dict(),
        "settings": settings.__dict__,
        "exact_feasible_starts": int(candidates["Exact_Both_Pass"].sum()),
        "safety_feasible_starts": int(candidates["Safety_Both_Pass"].sum()),
        "top_k_count": int(len(top_k)),
        "flexdc_validations": int(len(validation_df)),
        "actual_feasible_count": None if args.dry_run_flexdc else int(validation_df.get("Actual_Both_Pass", pd.Series(dtype=bool)).sum()),
        "selected_actual_feasible_candidate": selected,
    }
    write_json(out_dir / f"{prefix}_summary.json", summary)

    print("\nEnd-to-end status:", final_status)
    if not validation_df.empty:
        display_columns = [
            "Candidate_Label",
            "Pbar_kw_per_server",
            "R_kw_per_server",
            "Predicted_P90_Tracking",
            "Actual_P90_Tracking",
            "Predicted_Max_Pj",
            "Actual_Max_Pj",
            "Predicted_Full_Objective",
            "Actual_Full_Objective",
            "Actual_Both_Pass",
        ]
        available = [column for column in display_columns if column in validation_df.columns]
        print(validation_df[available].to_string(index=False))
    if selected is not None:
        print("\nSelected actual-feasible candidate:")
        for field in ["Candidate_Label", "Pbar_kw_per_server", "R_kw_per_server", "weights", "Actual_P90_Tracking", "Actual_Max_Pj", "Actual_Full_Objective"]:
            print(f"{field}: {selected.get(field)}")
    elif not args.dry_run_flexdc:
        print("\nNo FlexDC-validated feasible candidate was found. The script did not silently select an infeasible fallback.")
    print("\nOutputs:", out_dir)


if __name__ == "__main__":
    main()
