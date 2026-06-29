"""Optimize P, R, and workload weights using a frozen CONDOR/FlexDC model.

This is a minimal FlexDC-input adaptation of released CONDOR's
model_pr_descent(): freeze the model, differentiate a weighted sum of predicted
cost outputs, and update P, R, and workload weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import Softmax

from data_center_model import DataCenterModel
from am_condor_predict_one import (
    DEFAULT_COST_WEIGHTS,
    build_model_inputs,
    choose_device,
    inverse_scale_predictions,
    parse_float_list,
    predict_costs,
    read_workload_config,
    weighted_objective,
)


def calculate_pr_bounds(workload_config, pbar_lower_factor, pbar_upper_factor, pr_upper_factor, r_lower):
    jobs, _ = read_workload_config(workload_config)
    pmin = float(jobs[:, 0].min()) / 1000.0
    pmax = float(jobs[:, 1].max()) / 1000.0
    return {
        "Pmin_kw_per_server": pmin,
        "Pmax_kw_per_server": pmax,
        "Pbar_lower_bound_kw_per_server": float(pbar_lower_factor) * pmin,
        "Pbar_upper_bound_kw_per_server": float(pbar_upper_factor) * pmax,
        "PR_upper_bound_kw_per_server": float(pr_upper_factor) * pmax,
        "R_lower_bound_kw_per_server": float(r_lower),
    }


def calculate_weight_bounds(job_count: int, server_count: int):
    equal = 1.0 / job_count
    relative_lower = 0.1 * equal
    relative_upper = 4.0 * equal
    server_lower = 1.0 / server_count
    lower = max(relative_lower, server_lower)
    upper = min(relative_upper, 1.0 - (job_count - 1) * lower)
    return lower, upper


def project_weights_to_bounds(weights: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Project approximately to simplex with box bounds; simple and robust for J=4."""
    projected = torch.clamp(weights, lower, upper)
    for _ in range(30):
        diff = 1.0 - projected.sum()
        if torch.abs(diff) < 1e-8:
            break
        if diff > 0:
            free = projected < (upper - 1e-8)
        else:
            free = projected > (lower + 1e-8)
        if not bool(free.any()):
            break
        projected[free] += diff / free.sum()
        projected = torch.clamp(projected, lower, upper)
    return projected / projected.sum()


def optimize_inputs(
    model_file,
    workload_config,
    experiment_config,
    norm_source_results_csv,
    start_pbar_kw_per_server,
    start_r_kw_per_server,
    start_weights,
    cost_weights=DEFAULT_COST_WEIGHTS,
    iterations=150,
    lr=1e-2,
    device_name="auto",
    server_count_override=None,
    utilization_override=None,
    pbar_lower_factor=0.9,
    pbar_upper_factor=1.0,
    pr_upper_factor=1.2,
    r_lower=0.01,
):
    device = choose_device(device_name)
    initial = build_model_inputs(
        start_pbar_kw_per_server,
        start_r_kw_per_server,
        start_weights,
        workload_config,
        experiment_config,
        norm_source_results_csv,
        server_count_override=server_count_override,
        utilization_override=utilization_override,
    )
    server_count = int(initial["server_count"])
    utilization = float(initial["utilization"])
    job_count = int(initial["workload_mix_size"])
    pr_bounds = calculate_pr_bounds(workload_config, pbar_lower_factor, pbar_upper_factor, pr_upper_factor, r_lower)
    weight_lower, weight_upper = calculate_weight_bounds(job_count, server_count)

    model = DataCenterModel()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Optimize the same P/R input representation used by released CONDOR: p_norm/r_norm.
    p_ratio = torch.tensor(float(initial["Pbar_ratio"]), dtype=torch.float32, device=device, requires_grad=True)
    r_ratio = torch.tensor(float(initial["R_ratio"]), dtype=torch.float32, device=device, requires_grad=True)

    workload_unscaled = initial["workload_unscaled"]
    norm = initial["workload_norm_weights"]
    fixed_jobs = torch.tensor(workload_unscaled[:, :6] / norm[:6], dtype=torch.float32, device=device)
    weights = torch.tensor(np.asarray(start_weights, dtype=np.float32), dtype=torch.float32, device=device, requires_grad=True)
    softmax = Softmax(dim=0)

    records = []
    for iteration in range(iterations + 1):
        workload = torch.cat([fixed_jobs, weights.unsqueeze(1)], dim=1).unsqueeze(0)
        sim_features = torch.stack([
            p_ratio,
            r_ratio,
            torch.tensor(float(server_count), dtype=torch.float32, device=device),
            torch.tensor(float(utilization), dtype=torch.float32, device=device),
            torch.tensor(float(job_count), dtype=torch.float32, device=device),
        ])
        pred_scaled = model(sim_features, workload).reshape(-1)
        objective = sum(float(cost_weights[i]) * pred_scaled[i] for i in range(3))

        pbar = float(p_ratio.detach().cpu()) * initial["Pbar_denominator_watts"] / (1000.0 * server_count)
        reserve = float(r_ratio.detach().cpu()) * initial["R_denominator_watts"] / (1000.0 * server_count)
        pred_np = pred_scaled.detach().cpu().numpy()
        raw_np = inverse_scale_predictions(pred_np, server_count, job_count)
        records.append({
            "Iteration": iteration,
            "Pbar_kw_per_server": pbar,
            "R_kw_per_server": reserve,
            "Pbar_ratio": float(p_ratio.detach().cpu()),
            "R_ratio": float(r_ratio.detach().cpu()),
            **{f"Weight_{i}": float(weights[i].detach().cpu()) for i in range(job_count)},
            "Predicted_Scaled_cost_power": float(pred_np[0]),
            "Predicted_Scaled_cost_error": float(pred_np[1]),
            "Predicted_Scaled_cost_qos": float(pred_np[2]),
            "Predicted_Raw_cost_power": float(raw_np[0]),
            "Predicted_Raw_cost_error": float(raw_np[1]),
            "Predicted_Raw_cost_qos": float(raw_np[2]),
            "Predicted_Optimization_Objective": float(objective.detach().cpu()),
        })
        if iteration == iterations:
            break

        objective.backward()
        with torch.no_grad():
            p_ratio -= float(lr) * p_ratio.grad
            r_ratio -= float(lr) * r_ratio.grad
            weight_update = softmax(weights - weights.grad) - weights
            weights += float(lr) * weight_update
            weights.copy_(project_weights_to_bounds(weights, weight_lower, weight_upper))

            # Project P/R in physical kW/server units, then convert back to model ratios.
            pbar = float(p_ratio.detach().cpu()) * initial["Pbar_denominator_watts"] / (1000.0 * server_count)
            reserve = float(r_ratio.detach().cpu()) * initial["R_denominator_watts"] / (1000.0 * server_count)
            pbar = float(np.clip(pbar, pr_bounds["Pbar_lower_bound_kw_per_server"], pr_bounds["Pbar_upper_bound_kw_per_server"]))
            reserve_upper = min(pbar - 1e-6, pr_bounds["PR_upper_bound_kw_per_server"] - pbar)
            if reserve_upper < pr_bounds["R_lower_bound_kw_per_server"]:
                raise ValueError("No feasible R after projecting Pbar. Adjust bounds or starting point.")
            reserve = float(np.clip(reserve, pr_bounds["R_lower_bound_kw_per_server"], reserve_upper))
            p_ratio.copy_(torch.tensor(pbar * 1000.0 * server_count / initial["Pbar_denominator_watts"], dtype=torch.float32, device=device))
            r_ratio.copy_(torch.tensor(reserve * 1000.0 * server_count / initial["R_denominator_watts"], dtype=torch.float32, device=device))

        p_ratio.grad = None
        r_ratio.grad = None
        weights.grad = None

    trajectory = pd.DataFrame(records)
    best = trajectory.loc[trajectory["Predicted_Optimization_Objective"].idxmin()].to_dict()
    candidate_weights = [float(best[f"Weight_{i}"]) for i in range(job_count)]

    # Avoid recomputing the workload normalizer here; the first and best rows of
    # the trajectory already contain the exact model predictions used by descent.
    start_row = trajectory.iloc[0].to_dict()
    comparison = pd.DataFrame([
        {"Configuration": "Starting configuration", **start_row},
        {"Configuration": "CONDOR-selected configuration", **best},
    ])
    candidate_prediction = {
        key: float(best[key])
        for key in [
            "Predicted_Scaled_cost_power",
            "Predicted_Scaled_cost_error",
            "Predicted_Scaled_cost_qos",
            "Predicted_Raw_cost_power",
            "Predicted_Raw_cost_error",
            "Predicted_Raw_cost_qos",
            "Predicted_Optimization_Objective",
        ]
    }
    candidate = {
        "starting_pbar_kw_per_server": float(start_pbar_kw_per_server),
        "starting_r_kw_per_server": float(start_r_kw_per_server),
        "starting_weights": [float(x) for x in start_weights],
        "optimized_pbar_kw_per_server": float(best["Pbar_kw_per_server"]),
        "optimized_r_kw_per_server": float(best["R_kw_per_server"]),
        "optimized_weights": candidate_weights,
        "best_iteration": int(best["Iteration"]),
        "cost_weights": [float(x) for x in cost_weights],
        "predicted_optimization_objective": float(best["Predicted_Optimization_Objective"]),
        "prediction_at_candidate": candidate_prediction,
        "pr_bounds": pr_bounds,
        "weight_bounds": [float(weight_lower), float(weight_upper)],
        "device": str(device),
    }
    return trajectory, candidate, comparison


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


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize P/R/weights with a frozen CONDOR/FlexDC model.")
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
    parser.add_argument("--pbar-lower-factor", type=float, default=0.9)
    parser.add_argument("--pbar-upper-factor", type=float, default=1.0)
    parser.add_argument("--pr-upper-factor", type=float, default=1.2)
    parser.add_argument("--r-lower-kw-per-server", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", default="condor_optimize_one_output")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_weights = parse_float_list(args.start_weights, name="--start-weights")
    cost_weights = parse_float_list(args.cost_weights, expected_len=3, name="--cost-weights")

    run = init_wandb(args, {**vars(args), "cost_weights": cost_weights})
    trajectory, candidate, comparison = optimize_inputs(
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
    comparison.to_csv(out_dir / "optimization_comparison_before_validation.csv", index=False)
    with open(out_dir / "optimized_candidate.json", "w") as f:
        json.dump(candidate, f, indent=2)

    if run is not None:
        for _, row in trajectory.iterrows():
            run.log({
                "optimization/objective": row["Predicted_Optimization_Objective"],
                "optimization/pbar_kw_per_server": row["Pbar_kw_per_server"],
                "optimization/r_kw_per_server": row["R_kw_per_server"],
            }, step=int(row["Iteration"]))
        run.summary["best_iteration"] = candidate["best_iteration"]
        run.summary["optimized_objective"] = candidate["predicted_optimization_objective"]
        run.finish()

    print(comparison[["Configuration", "Pbar_kw_per_server", "R_kw_per_server", "Predicted_Optimization_Objective"]].to_string(index=False))
    print("\nSaved:", out_dir / "optimization_trajectory.csv")
    print("Saved:", out_dir / "optimization_comparison_before_validation.csv")
    print("Saved:", out_dir / "optimized_candidate.json")


if __name__ == "__main__":
    main()
