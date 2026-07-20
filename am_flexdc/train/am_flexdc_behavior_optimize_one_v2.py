#!/usr/bin/env python3
"""Multi-start gradient optimization through the frozen CONDOR behavior model v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from am_flexdc_behavior_inference_utilities_v2 import (
    OptimizationSettings,
    calculate_pr_bounds,
    dataframe_for_csv,
    load_behavior_model,
    optimize_candidates,
    read_experiment_config,
    read_workload_config,
    resolve_safety_limits,
    resolve_effective_weight_bounds,
    score_candidate_table_with_checkpoint,
    write_json,
)


def parse_weights(text: str | None) -> list[float] | None:
    if text is None:
        return None
    return [float(piece.strip()) for piece in text.split(",") if piece.strip()]


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
    parser.add_argument(
        "--enforce-flexdc-weight-bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce the same server/job-count weight bounds as the FlexDC wizard (default: true)",
    )
    parser.add_argument("--weight-min-fraction-of-equal", type=float, default=0.1)
    parser.add_argument("--weight-max-multiple-of-equal", type=float, default=4.0)
    parser.add_argument("--weight-min", type=float, default=None, help="Optional stricter absolute lower bound")
    parser.add_argument("--weight-max", type=float, default=None, help="Optional stricter absolute upper bound")

    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-prefix", default="behavior_v2_optimization")
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
        enforce_flexdc_weight_bounds=args.enforce_flexdc_weight_bounds,
        weight_min_fraction_of_equal=args.weight_min_fraction_of_equal,
        weight_max_multiple_of_equal=args.weight_max_multiple_of_equal,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        r_over_p_max=args.r_over_p_max,
        pbar_min_override=args.pbar_min,
        pbar_max_override=args.pbar_max,
        r_min_override=args.r_min,
        r_max_override=args.r_max,
        log_every=args.log_every,
    )

    effective_weight_bounds = resolve_effective_weight_bounds(
        settings,
        job_count=workload.job_count,
        server_count=experiment.server_count,
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

    prefix = args.output_prefix
    all_path = out_dir / f"{prefix}_all_starts.csv"
    top_path = out_dir / f"{prefix}_top_k.csv"
    trajectory_path = out_dir / f"{prefix}_trajectory.csv"
    metadata_path = out_dir / f"{prefix}_metadata.json"
    dataframe_for_csv(candidates).to_csv(all_path, index=False)
    dataframe_for_csv(top_k).to_csv(top_path, index=False)
    dataframe_for_csv(trajectory).to_csv(trajectory_path, index=False)
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(loaded.checkpoint.get("epoch", -1)),
        "workload": asdict_safe(workload),
        "experiment": asdict_safe(experiment),
        "bounds": bounds.to_dict(),
        "safety": safety.to_dict(),
        "settings": settings.__dict__,
        "effective_weight_bounds": effective_weight_bounds.to_dict(),
        "predicted_safety_feasible_starts": int(candidates["Safety_Both_Pass"].sum()),
        "predicted_exact_feasible_starts": int(candidates["Exact_Both_Pass"].sum()),
        "top_k_count": int(len(top_k)),
        "status": "predicted_feasible_candidates_found" if len(top_k) else "no_predicted_feasible_candidate",
        "outputs": {
            "all_starts": str(all_path),
            "top_k": str(top_path),
            "trajectory": str(trajectory_path),
        },
    }
    write_json(metadata_path, metadata)

    print("\nOptimization status:", metadata["status"])
    print("Exact-feasible starts:", metadata["predicted_exact_feasible_starts"])
    print("Safety-feasible starts:", metadata["predicted_safety_feasible_starts"])
    print("Distinct top-k candidates:", len(top_k))
    if len(top_k):
        columns = [
            "Candidate_Rank",
            "Pbar_kw_per_server",
            "R_kw_per_server",
            "weights",
            "Predicted_P90_Tracking",
            "Predicted_Max_Pj",
            "Predicted_M_RSR",
            "Predicted_Full_Objective",
            "Safety_Tracking_Slack",
            "Safety_QoS_Slack",
        ]
        print(top_k[columns].to_string(index=False))
    else:
        print("No candidate met the configured selection thresholds. The script did not silently substitute an unconstrained candidate.")
    print("\nSaved outputs under:", out_dir)


def asdict_safe(value) -> dict:
    if hasattr(value, "__dataclass_fields__"):
        result = {}
        for key in value.__dataclass_fields__:
            item = getattr(value, key)
            if hasattr(item, "tolist"):
                item = item.tolist()
            result[key] = item
        return result
    raise TypeError(type(value))


if __name__ == "__main__":
    main()
