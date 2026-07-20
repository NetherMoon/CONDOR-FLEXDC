#!/usr/bin/env python3
"""Predict one FlexDC configuration with the CONDOR behavior model v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from am_flexdc_behavior_inference_utilities_v2 import (
    calculate_pr_bounds,
    calculate_weight_bounds,
    dataframe_for_csv,
    load_behavior_model,
    predict_configuration,
    read_experiment_config,
    read_workload_config,
    resolve_safety_limits,
    validate_weight_bounds,
    write_json,
)


def parse_weights(text: str) -> list[float]:
    return [float(piece.strip()) for piece in text.split(",") if piece.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--pbar", type=float, required=True, help="Pbar in kW/server")
    parser.add_argument("--r", type=float, required=True, help="R in kW/server")
    parser.add_argument("--weights", required=True, help="Comma-separated AQA weights")
    parser.add_argument("--server-count", type=int, default=None)
    parser.add_argument("--utilization", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tracking-limit", type=float, default=None, help="Absolute predicted tracking limit")
    parser.add_argument("--qos-limit", type=float, default=None, help="Absolute predicted per-job QoS limit")
    parser.add_argument("--tracking-margin", type=float, default=0.04)
    parser.add_argument("--qos-margin", type=float, default=0.01)
    parser.add_argument("--pbar-lower-factor", type=float, default=0.9)
    parser.add_argument("--pbar-upper-factor", type=float, default=1.0)
    parser.add_argument("--pr-upper-factor", type=float, default=1.2)
    parser.add_argument("--r-lower", type=float, default=0.01)
    parser.add_argument("--r-over-p-max", type=float, default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-job-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    weight_values = parse_weights(args.weights)
    weight_bounds = calculate_weight_bounds(workload.job_count, experiment.server_count)
    validate_weight_bounds(weight_values, weight_bounds)

    result, per_job = predict_configuration(
        loaded,
        workload=workload,
        experiment=experiment,
        pbar_kw_per_server=args.pbar,
        r_kw_per_server=args.r,
        weights=weight_values,
        safety=safety,
        bounds=bounds,
        r_over_p_max=args.r_over_p_max,
    )

    result["Weight_Bounds"] = weight_bounds.to_dict()
    result["Weight_Bounds_Pass"] = True

    print("\nPrediction summary")
    display_fields = [
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "weights",
        "Predicted_Mean_Tracking",
        "Predicted_P90_Tracking",
        "Predicted_Max_Pj",
        "Predicted_M_RSR",
        "Predicted_Ctrack",
        "Predicted_CQoS",
        "Predicted_Full_Objective",
        "Exact_Both_Pass",
        "Safety_Both_Pass",
        "Safety_Tracking_Slack",
        "Safety_QoS_Slack",
    ]
    for field in display_fields:
        print(f"{field}: {result[field]}")
    print("\nPer-job QoS")
    print(per_job.to_string(index=False))

    if args.out_json:
        write_json(args.out_json, {"summary": result, "per_job": per_job.to_dict(orient="records")})
        print("Saved:", Path(args.out_json).resolve())
    if args.out_job_csv:
        dataframe_for_csv(per_job).to_csv(args.out_job_csv, index=False)
        print("Saved:", Path(args.out_job_csv).resolve())


if __name__ == "__main__":
    main()
