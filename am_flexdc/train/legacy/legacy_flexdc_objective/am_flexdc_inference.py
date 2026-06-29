"""
Minimal FlexDC adaptation of CONDOR's inference_scripts.py.

The frozen model is unchanged. This script changes only:
- legacy workload-dictionary loading -> FlexDC results/diagnostics context
- legacy weighted objective -> M_RSR + C_track + C_Qos
- unconstrained P/R and weights -> current pilot feasibility bounds
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import Softmax

from data_center_model import DataCenterModel


WORKLOAD_NORMALIZER = np.array([1000, 1000, 3600, 3600, 1, 1, 1], dtype=np.float32)
JOIN_KEYS = ["Source_Output_Dir", "Iteration"]


def load_context(results_csv, diagnostics_csv, workload, server_count, utilization, weight_sample_id):
    results = pd.read_csv(results_csv)
    diagnostics = pd.read_csv(diagnostics_csv)

    diagnostic_columns = JOIN_KEYS + [
        "Ctrack_Weighted_Cost",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "Diagnostic_FullPaperObjective_Cost",
        "Pbar_lower_bound_kw_per_server",
        "Pbar_upper_bound_kw_per_server",
        "PR_upper_bound_kw_per_server",
        "R_lower_bound_kw_per_server",
        "Weight_Final_Lower_Bound",
        "Weight_Final_Upper_Bound",
    ]
    data = results.merge(
        diagnostics[diagnostic_columns],
        on=JOIN_KEYS,
        how="inner",
        validate="one_to_one",
    )
    all_context = data[
        (data["Workload_Name"] == workload)
        & (data["server_count"].astype(int) == int(server_count))
        & np.isclose(data["utilization"].astype(float), float(utilization))
    ].copy()
    data = all_context[all_context["Weight_Sample_ID"].astype(int) == int(weight_sample_id)].copy()
    if data.empty:
        raise ValueError("No rows match the requested workload/server/utilization/weight context.")

    weight_columns = sorted(
        [name for name in data.columns if name.startswith("Weight_") and name.split("_")[-1].isdigit()],
        key=lambda name: int(name.split("_")[-1]),
    )
    first = data.iloc[0]
    jobs = np.asarray(ast.literal_eval(first["workload_mix"]), dtype=np.float32)
    weights = first[weight_columns].to_numpy(dtype=np.float32)

    if jobs.ndim != 2 or jobs.shape != (len(weights), 6):
        raise ValueError("Expected workload_mix to be J x 6 and Weight_0...Weight_(J-1).")

    bound_columns = [
        "Pbar_lower_bound_kw_per_server",
        "Pbar_upper_bound_kw_per_server",
        "PR_upper_bound_kw_per_server",
        "R_lower_bound_kw_per_server",
        "Weight_Final_Lower_Bound",
        "Weight_Final_Upper_Bound",
    ]
    if any(data[column].nunique() != 1 for column in bound_columns):
        raise ValueError("Pilot feasibility bounds are not constant in this context.")

    # Start from a real middle-grid pilot row, not an invented P/R point.
    p_values = np.sort(data["Pbar_kw_per_server"].unique())
    p_start = float(p_values[len(p_values) // 2])
    p_rows = data[np.isclose(data["Pbar_kw_per_server"], p_start)]
    r_values = np.sort(p_rows["R_kw_per_server"].unique())
    r_start = float(r_values[len(r_values) // 2])
    start = p_rows[np.isclose(p_rows["R_kw_per_server"], r_start)].iloc[0]

    return {
        "jobs": jobs,
        "weights": weights,
        "start": start,
        "best_sampled": data.loc[data["Diagnostic_FullPaperObjective_Cost"].idxmin()],
        "best_sampled_all": all_context.loc[all_context["Diagnostic_FullPaperObjective_Cost"].idxmin()],
        "weight_columns": weight_columns,
        "p_lower": float(first["Pbar_lower_bound_kw_per_server"]),
        "p_upper": float(first["Pbar_upper_bound_kw_per_server"]),
        "pr_upper": float(first["PR_upper_bound_kw_per_server"]),
        "r_lower": float(first["R_lower_bound_kw_per_server"]),
        "weight_lower": float(first["Weight_Final_Lower_Bound"]),
        "weight_upper": float(first["Weight_Final_Upper_Bound"]),
    }


def predict_components(model, pbar, reserve, jobs, weights, server_count, utilization):
    features = torch.tensor(
        [pbar, reserve, server_count, utilization, len(jobs)],
        dtype=torch.float32,
    )
    workload = torch.tensor(
        np.column_stack([jobs, weights]) / WORKLOAD_NORMALIZER,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        return model(features, workload).detach().cpu().numpy().reshape(-1)


def model_pr_descent(p_init,
                     r_init,
                     workload_mix,
                     model,
                     client_count=1000,
                     util=0.60,
                     p_lower=None,
                     p_upper=None,
                     pr_upper=None,
                     r_lower=0.01,
                     weight_lower=0.025,
                     weight_upper=0.925,
                     iterations=150,
                     lr=1e-3):
    # This retains the original function's manual gradient-descent structure.
    jobs = np.asarray(workload_mix[:, :6], dtype=np.float32)
    initial_weights = np.asarray(workload_mix[:, 6], dtype=np.float32)
    job_count = len(jobs)

    if p_lower is None or p_upper is None or pr_upper is None:
        raise ValueError("P/R feasibility bounds are required.")
    if not np.isclose(initial_weights.sum(), 1.0):
        raise ValueError("Initial weights must sum to 1.")

    free_weight_mass = 1.0 - job_count * weight_lower
    if free_weight_mass <= 0 or weight_upper + 1e-6 < weight_lower + free_weight_mass:
        raise ValueError("Weight bounds are incompatible with the lower-bounded simplex.")

    # lower + free_mass * Softmax(logits) enforces sum(w)=1 and pilot weight bounds.
    starting_simplex = (initial_weights - weight_lower) / free_weight_mass
    if np.any(starting_simplex <= 0):
        raise ValueError("The starting weights are on the lower bound; use an interior vector.")

    jobs = torch.tensor(jobs / WORKLOAD_NORMALIZER[:6], dtype=torch.float32)
    pbar = torch.tensor(float(p_init), dtype=torch.float32, requires_grad=True)
    reserve = torch.tensor(float(r_init), dtype=torch.float32, requires_grad=True)
    logits = torch.tensor(np.log(starting_simplex), dtype=torch.float32, requires_grad=True)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records = []
    for iteration in range(iterations + 1):
        weights = weight_lower + free_weight_mass * Softmax(dim=0)(logits)
        workload = torch.cat([jobs, weights.unsqueeze(1)], dim=1).unsqueeze(0)
        sim_config = torch.stack([
            pbar,
            reserve,
            torch.tensor(float(client_count)),
            torch.tensor(float(util)),
            torch.tensor(float(job_count)),
        ])

        prediction = model(sim_config, workload).reshape(-1)
        objective = prediction.sum()

        records.append({
            "Iteration": iteration,
            "Pbar_kw_per_server": float(pbar.detach()),
            "R_kw_per_server": float(reserve.detach()),
            **{f"Weight_{i}": float(weights[i].detach()) for i in range(job_count)},
            "Predicted_M_RSR": float(prediction[0].detach()),
            "Predicted_C_track": float(prediction[1].detach()),
            "Predicted_C_Qos": float(prediction[2].detach()),
            "Predicted_C_total": float(objective.detach()),
        })

        if iteration == iterations:
            break

        objective.backward()
        with torch.no_grad():
            pbar -= lr * pbar.grad
            reserve -= lr * reserve.grad
            logits -= lr * logits.grad

            pbar.clamp_(p_lower, p_upper)
            reserve.clamp_(r_lower, min(float(pbar) - 1e-6, pr_upper - float(pbar)))
        pbar.grad = reserve.grad = logits.grad = None

    return pd.DataFrame(records)


def actual_values(row):
    return {
        "Actual_M_RSR": float(row["Simulator_RSR_Total_Cost"]),
        "Actual_C_track": float(row["Ctrack_Weighted_Cost"]),
        "Actual_C_Qos": float(row["Diagnostic_FlexDC_SoftPlus_QoS_Cost"]),
        "Actual_C_total": float(row["Diagnostic_FullPaperObjective_Cost"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize the frozen FlexDC-objective model.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--out-dir", default="flexdc_inference_results")
    parser.add_argument("--workload", default="W1-train")
    parser.add_argument("--server-count", type=int, default=1000)
    parser.add_argument("--utilization", type=float, default=0.60)
    parser.add_argument("--weight-sample-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context = load_context(
        args.results_csv,
        args.diagnostics_csv,
        args.workload,
        args.server_count,
        args.utilization,
        args.weight_sample_id,
    )

    model = DataCenterModel()
    model.load_state_dict(torch.load(args.model_file, map_location="cpu"))

    start = context["start"]
    start_workload = np.column_stack([context["jobs"], context["weights"]])
    trajectory = model_pr_descent(
        float(start["Pbar_kw_per_server"]),
        float(start["R_kw_per_server"]),
        start_workload,
        model,
        client_count=args.server_count,
        util=args.utilization,
        p_lower=context["p_lower"],
        p_upper=context["p_upper"],
        pr_upper=context["pr_upper"],
        r_lower=context["r_lower"],
        weight_lower=context["weight_lower"],
        weight_upper=context["weight_upper"],
        iterations=args.iterations,
        lr=args.lr,
    )
    trajectory.to_csv(out_dir / "optimization_trajectory.csv", index=False)

    start_prediction = trajectory.iloc[0]
    optimized = trajectory.loc[trajectory["Predicted_C_total"].idxmin()]
    best_sampled = context["best_sampled"]
    best_prediction = predict_components(
        model,
        float(best_sampled["Pbar_kw_per_server"]),
        float(best_sampled["R_kw_per_server"]),
        context["jobs"],
        context["weights"],
        args.server_count,
        args.utilization,
    )
    best_sampled_all = context["best_sampled_all"]
    best_all_weights = best_sampled_all[context["weight_columns"]].to_numpy(dtype=float)
    best_all_prediction = predict_components(
        model,
        float(best_sampled_all["Pbar_kw_per_server"]),
        float(best_sampled_all["R_kw_per_server"]),
        context["jobs"],
        best_all_weights,
        args.server_count,
        args.utilization,
    )

    weight_names = [f"Weight_{i}" for i in range(len(context["weights"]))]
    comparison = pd.DataFrame([
        {
            "Configuration": "Starting sampled pilot point",
            "Weight_Sample_ID": int(args.weight_sample_id),
            "Pbar_kw_per_server": float(start["Pbar_kw_per_server"]),
            "R_kw_per_server": float(start["R_kw_per_server"]),
            **dict(zip(weight_names, context["weights"])),
            **{name: float(start_prediction[name]) for name in [
                "Predicted_M_RSR", "Predicted_C_track", "Predicted_C_Qos", "Predicted_C_total"
            ]},
            **actual_values(start),
        },
        {
            "Configuration": "Surrogate optimized point (run in FlexDC next)",
            "Weight_Sample_ID": "continuous",
            "Pbar_kw_per_server": float(optimized["Pbar_kw_per_server"]),
            "R_kw_per_server": float(optimized["R_kw_per_server"]),
            **{name: float(optimized[name]) for name in weight_names},
            **{name: float(optimized[name]) for name in [
                "Predicted_M_RSR", "Predicted_C_track", "Predicted_C_Qos", "Predicted_C_total"
            ]},
        },
        {
            "Configuration": "Best sampled equal-weight pilot point",
            "Weight_Sample_ID": int(args.weight_sample_id),
            "Pbar_kw_per_server": float(best_sampled["Pbar_kw_per_server"]),
            "R_kw_per_server": float(best_sampled["R_kw_per_server"]),
            **dict(zip(weight_names, context["weights"])),
            "Predicted_M_RSR": float(best_prediction[0]),
            "Predicted_C_track": float(best_prediction[1]),
            "Predicted_C_Qos": float(best_prediction[2]),
            "Predicted_C_total": float(best_prediction.sum()),
            **actual_values(best_sampled),
        },
        {
            "Configuration": "Best sampled pilot point (all accepted weights)",
            "Weight_Sample_ID": int(best_sampled_all["Weight_Sample_ID"]),
            "Pbar_kw_per_server": float(best_sampled_all["Pbar_kw_per_server"]),
            "R_kw_per_server": float(best_sampled_all["R_kw_per_server"]),
            **dict(zip(weight_names, best_all_weights)),
            "Predicted_M_RSR": float(best_all_prediction[0]),
            "Predicted_C_track": float(best_all_prediction[1]),
            "Predicted_C_Qos": float(best_all_prediction[2]),
            "Predicted_C_total": float(best_all_prediction.sum()),
            **actual_values(best_sampled_all),
        },
    ])
    comparison.to_csv(out_dir / "optimization_comparison_before_validation.csv", index=False)

    summary = {
        "workload": args.workload,
        "server_count": args.server_count,
        "utilization": args.utilization,
        "starting_weight_sample_id": args.weight_sample_id,
        "optimized_pbar_kw_per_server": float(optimized["Pbar_kw_per_server"]),
        "optimized_r_kw_per_server": float(optimized["R_kw_per_server"]),
        "optimized_weights": [float(optimized[name]) for name in weight_names],
        "predicted_M_RSR": float(optimized["Predicted_M_RSR"]),
        "predicted_C_track": float(optimized["Predicted_C_track"]),
        "predicted_C_Qos": float(optimized["Predicted_C_Qos"]),
        "predicted_C_total": float(optimized["Predicted_C_total"]),
        "pbar_bounds_kw_per_server": [context["p_lower"], context["p_upper"]],
        "pr_upper_bound_kw_per_server": context["pr_upper"],
        "r_lower_bound_kw_per_server": context["r_lower"],
        "weight_bounds": [context["weight_lower"], context["weight_upper"]],
    }
    with open(out_dir / "optimized_candidate.json", "w") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'optimization_trajectory.csv'}")
    print(f"Saved: {out_dir / 'optimization_comparison_before_validation.csv'}")
    print(f"Saved: {out_dir / 'optimized_candidate.json'}")


if __name__ == "__main__":
    main()
