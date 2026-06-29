"""Optimize P, R, and workload weights with a frozen unified surrogate.

This follows the structure of CONDOR's original model_pr_descent(): keep the
neural model fixed, compute the gradient of a weighted sum of predicted outputs,
and update only P, R, and workload weights.

The same file works for the four trained variants:
    condor/normal, condor/raw, flexdc/normal, flexdc/raw
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
from am_unified_predict_one import (
    build_model_inputs,
    choose_device,
    default_objective_weights,
    inverse_scale_targets,
    parse_bool_text,
    parse_float_list,
    parse_objective_weights,
    predict_costs,
    read_workload_config,
    resolve_use_norm_cost,
    target_names,
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
    # Same feasibility rules used by the FlexDC data wizard's weight generator.
    equal = 1.0 / job_count
    relative_lower = 0.1 * equal
    relative_upper = 4.0 * equal
    server_lower = 1.0 / server_count
    lower = max(relative_lower, server_lower)
    upper = min(relative_upper, 1.0 - (job_count - 1) * lower)
    if upper < lower:
        raise ValueError(f"Infeasible weight bounds: lower={lower}, upper={upper}")
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


def model_feature_to_physical(feature_value: float, denominator_watts: float, server_count: int, use_norm_pr: bool) -> float:
    if use_norm_pr:
        return float(feature_value) * float(denominator_watts) / (1000.0 * float(server_count))
    return float(feature_value)


def physical_to_model_feature(kw_per_server: float, denominator_watts: float, server_count: int, use_norm_pr: bool) -> float:
    if use_norm_pr:
        return float(kw_per_server) * 1000.0 * float(server_count) / float(denominator_watts)
    return float(kw_per_server)


def optimize_inputs(
    model_file,
    workload_config,
    experiment_config,
    norm_source_results_csv,
    start_pbar_kw_per_server,
    start_r_kw_per_server,
    start_weights,
    target_family="condor",
    target_mode="normal",
    raw_qos_aggregation="mean",
    use_norm_cost=None,
    use_norm_pr=True,
    objective_weights=None,
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
    target_family = target_family.lower().strip()
    target_mode = target_mode.lower().strip()
    raw_qos_aggregation = raw_qos_aggregation.lower().strip()
    use_norm_cost = resolve_use_norm_cost(target_family, use_norm_cost)
    if objective_weights is None:
        objective_weights = default_objective_weights(target_family)
    names = target_names(target_family, target_mode, raw_qos_aggregation)

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
        use_norm_pr=use_norm_pr,
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

    if use_norm_pr:
        p_init_feature = float(initial["Pbar_ratio"])
        r_init_feature = float(initial["R_ratio"])
    else:
        p_init_feature = float(start_pbar_kw_per_server)
        r_init_feature = float(start_r_kw_per_server)

    p_feature = torch.tensor(p_init_feature, dtype=torch.float32, device=device, requires_grad=True)
    r_feature = torch.tensor(r_init_feature, dtype=torch.float32, device=device, requires_grad=True)

    workload_unscaled = initial["workload_unscaled"]
    norm = initial["workload_norm_weights"]
    fixed_jobs = torch.tensor(workload_unscaled[:, :6] / norm[:6], dtype=torch.float32, device=device)
    weights = torch.tensor(np.asarray(start_weights, dtype=np.float32), dtype=torch.float32, device=device, requires_grad=True)
    softmax = Softmax(dim=0)

    records = []
    for iteration in range(iterations + 1):
        workload = torch.cat([fixed_jobs, weights.unsqueeze(1)], dim=1).unsqueeze(0)
        sim_features = torch.stack([
            p_feature,
            r_feature,
            torch.tensor(float(server_count), dtype=torch.float32, device=device),
            torch.tensor(float(utilization), dtype=torch.float32, device=device),
            torch.tensor(float(job_count), dtype=torch.float32, device=device),
        ])
        pred_targets = model(sim_features, workload).reshape(-1)
        objective = sum(float(objective_weights[i]) * pred_targets[i] for i in range(3))

        pbar = model_feature_to_physical(
            float(p_feature.detach().cpu()), initial["Pbar_denominator_watts"], server_count, use_norm_pr
        )
        reserve = model_feature_to_physical(
            float(r_feature.detach().cpu()), initial["R_denominator_watts"], server_count, use_norm_pr
        )
        pred_np = pred_targets.detach().cpu().numpy()
        unscaled_np, _ = inverse_scale_targets(
            pred_np,
            target_family,
            target_mode,
            use_norm_cost,
            raw_qos_aggregation,
            server_count,
            job_count,
        )

        row = {
            "Iteration": iteration,
            "Pbar_kw_per_server": pbar,
            "R_kw_per_server": reserve,
            "Pbar_feature": float(p_feature.detach().cpu()),
            "R_feature": float(r_feature.detach().cpu()),
            "Pbar_ratio": pbar * 1000.0 * server_count / initial["Pbar_denominator_watts"],
            "R_ratio": reserve * 1000.0 * server_count / initial["R_denominator_watts"],
            **{f"Weight_{i}": float(weights[i].detach().cpu()) for i in range(job_count)},
            "Predicted_Optimization_Objective": float(objective.detach().cpu()),
            "Predicted_Target_Sum": float(np.sum(pred_np)),
        }
        for idx, name in enumerate(names):
            row[f"Predicted_{name}"] = float(pred_np[idx])
            row[f"Predicted_unscaled_{name}"] = float(unscaled_np[idx])
        records.append(row)

        if iteration == iterations:
            break

        objective.backward()
        with torch.no_grad():
            p_feature -= float(lr) * p_feature.grad
            r_feature -= float(lr) * r_feature.grad
            weight_update = softmax(weights - weights.grad) - weights
            weights += float(lr) * weight_update
            weights.copy_(project_weights_to_bounds(weights, weight_lower, weight_upper))

            # Project P/R in physical kW/server units, then convert back to the
            # model's input representation. This keeps optimization inside the
            # FlexDC training/validation domain generated by the data wizard.
            pbar = model_feature_to_physical(
                float(p_feature.detach().cpu()), initial["Pbar_denominator_watts"], server_count, use_norm_pr
            )
            reserve = model_feature_to_physical(
                float(r_feature.detach().cpu()), initial["R_denominator_watts"], server_count, use_norm_pr
            )
            pbar = float(np.clip(pbar, pr_bounds["Pbar_lower_bound_kw_per_server"], pr_bounds["Pbar_upper_bound_kw_per_server"]))
            reserve_upper = min(pbar - 1e-6, pr_bounds["PR_upper_bound_kw_per_server"] - pbar)
            if reserve_upper < pr_bounds["R_lower_bound_kw_per_server"]:
                raise ValueError("No feasible R after projecting Pbar. Adjust bounds or starting point.")
            reserve = float(np.clip(reserve, pr_bounds["R_lower_bound_kw_per_server"], reserve_upper))
            p_feature.copy_(torch.tensor(
                physical_to_model_feature(pbar, initial["Pbar_denominator_watts"], server_count, use_norm_pr),
                dtype=torch.float32,
                device=device,
            ))
            r_feature.copy_(torch.tensor(
                physical_to_model_feature(reserve, initial["R_denominator_watts"], server_count, use_norm_pr),
                dtype=torch.float32,
                device=device,
            ))

        p_feature.grad = None
        r_feature.grad = None
        weights.grad = None

    trajectory = pd.DataFrame(records)
    best = trajectory.loc[trajectory["Predicted_Optimization_Objective"].idxmin()].to_dict()
    candidate_weights = [float(best[f"Weight_{i}"]) for i in range(job_count)]

    start_row = trajectory.iloc[0].to_dict()
    comparison = pd.DataFrame([
        {"Configuration": "Starting configuration", **start_row},
        {"Configuration": "Selected configuration", **best},
    ])
    candidate_prediction = {
        key: float(best[key])
        for key in best.keys()
        if key.startswith("Predicted_")
    }
    candidate = {
        "target_family": target_family,
        "target_mode": target_mode,
        "use_norm_cost": bool(use_norm_cost),
        "use_norm_pr": bool(use_norm_pr),
        "raw_qos_aggregation": raw_qos_aggregation,
        "target_names": names,
        "starting_pbar_kw_per_server": float(start_pbar_kw_per_server),
        "starting_r_kw_per_server": float(start_r_kw_per_server),
        "starting_weights": [float(x) for x in start_weights],
        "optimized_pbar_kw_per_server": float(best["Pbar_kw_per_server"]),
        "optimized_r_kw_per_server": float(best["R_kw_per_server"]),
        "optimized_weights": candidate_weights,
        "best_iteration": int(best["Iteration"]),
        "objective_weights": [float(x) for x in objective_weights],
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
    parser = argparse.ArgumentParser(description="Optimize P/R/weights with a frozen unified CONDOR/FlexDC model.")
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
    parser.add_argument("--pbar-lower-factor", type=float, default=0.9)
    parser.add_argument("--pbar-upper-factor", type=float, default=1.0)
    parser.add_argument("--pr-upper-factor", type=float, default=1.2)
    parser.add_argument("--r-lower-kw-per-server", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", default="unified_optimize_one_output")
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
    use_norm_cost = resolve_use_norm_cost(args.target_family, args.use_norm_cost)
    use_norm_pr = parse_bool_text(args.use_norm_pr, name="--use-norm-pr")
    objective_weights = parse_objective_weights(args.objective_weights, args.target_family)

    run = init_wandb(args, {**vars(args), "objective_weights": objective_weights, "use_norm_cost_resolved": use_norm_cost})
    trajectory, candidate, comparison = optimize_inputs(
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
