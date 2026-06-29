"""CONDOR input-gradient inference using FlexDC-generated data.

This is a minimal adaptation of the released CONDOR inference script:
- the frozen DataCenterModel and three-output semantics are unchanged;
- the released example cost weights default to [0.05, 0.7, 2.0];
- FlexDC CSV rows supply the workload/configuration inputs;
- P/R are kept inside the sampled FlexDC pilot domain so the candidate can be
  validated by the FlexDC simulator.

The model outputs the scaled targets used by the released CONDOR loader:
    y_power = 120 * cost_power / server_count
    y_error = 200 * cost_error / server_count
    y_qos   = cost_qos / workload_mix_size
The optimization objective is the released-style weighted sum of these outputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import Softmax

from data_center_model import DataCenterModel
from am_condor_flexdc_training_utilities import (
    CONDOR_POWER_COST_COEFFICIENT,
    CONDOR_QOS_BETA,
    CONDOR_QOS_RHO,
    CONDOR_QOS_THRESHOLD,
    DCDataset,
)


JOIN_KEYS = ["Source_Output_Dir", "Iteration"]
DEFAULT_COST_WEIGHTS = [0.05, 0.7, 2.0]


def condor_labels_from_row(row):
    """Reconstruct the same raw/scaled CONDOR labels used during training."""
    server_count = float(row["server_count"])
    workload_size = float(row["workload_mix_size"])

    raw_power = CONDOR_POWER_COST_COEFFICIENT * (
        float(row["P_actual_watts"]) - float(row["R_actual_watts"])
    )
    raw_error = float(row["Mtrack_Error_MeanAbs_Watts"]) / 1000.0
    probabilities = np.asarray(json.loads(row["QoS_Delay_Probabilities"]), dtype=float)
    raw_qos = CONDOR_QOS_BETA * np.logaddexp(
        0,
        CONDOR_QOS_RHO * (probabilities - CONDOR_QOS_THRESHOLD),
    ).sum()

    scaled = np.asarray([
        raw_power * 120.0 / server_count,
        raw_error * 200.0 / server_count,
        raw_qos / workload_size,
    ])
    raw = np.asarray([raw_power, raw_error, raw_qos])
    return raw, scaled


def inverse_scale_predictions(prediction, server_count, workload_size):
    """Convert model outputs from released-loader scaling back to raw components."""
    prediction = np.asarray(prediction, dtype=float)
    return np.asarray([
        prediction[0] * server_count / 120.0,
        prediction[1] * server_count / 200.0,
        prediction[2] * workload_size,
    ])


def load_context(results_csv, diagnostics_csv, workload, server_count, utilization, weight_sample_id,
                 start_pbar=None, start_r=None):
    results = pd.read_csv(results_csv)
    diagnostics = pd.read_csv(diagnostics_csv)

    diagnostic_columns = JOIN_KEYS + [
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

    context_all_weights = data[
        (data["Workload_Name"] == workload)
        & (data["server_count"].astype(int) == int(server_count))
        & np.isclose(data["utilization"].astype(float), float(utilization))
    ].copy()
    context = context_all_weights[
        context_all_weights["Weight_Sample_ID"].astype(int) == int(weight_sample_id)
    ].copy()
    if context.empty:
        raise ValueError("No pilot rows match the requested context.")

    weight_columns = sorted(
        [column for column in results.columns if column.startswith("Weight_") and column != "Weight_Sample_ID"],
        key=lambda column: int(column.split("_")[-1]),
    )
    first = context.iloc[0]
    jobs = np.asarray(json.loads(first["workload_mix"]), dtype=np.float32)
    initial_weights = first[weight_columns].to_numpy(dtype=np.float32)
    if jobs.ndim != 2 or jobs.shape != (len(initial_weights), 6):
        raise ValueError("Expected a J x 6 workload profile and J workload weights.")

    constant_columns = [
        "Pbar_denominator_watts",
        "R_denominator_watts",
        "Pbar_lower_bound_kw_per_server",
        "Pbar_upper_bound_kw_per_server",
        "PR_upper_bound_kw_per_server",
        "R_lower_bound_kw_per_server",
        "Weight_Final_Lower_Bound",
        "Weight_Final_Upper_Bound",
    ]
    for column in constant_columns:
        if context[column].nunique() != 1:
            raise ValueError(f"{column} is not constant in the selected context.")

    if start_pbar is None and start_r is None:
        # Start from a real central sampled point, not an invented configuration.
        p_values = np.sort(context["Pbar_kw_per_server"].unique())
        selected_p = float(p_values[len(p_values) // 2])
        p_rows = context[np.isclose(context["Pbar_kw_per_server"], selected_p)]
        r_values = np.sort(p_rows["R_kw_per_server"].unique())
        selected_r = float(r_values[len(r_values) // 2])
        start = p_rows[np.isclose(p_rows["R_kw_per_server"], selected_r)].iloc[0]
    elif start_pbar is not None and start_r is not None:
        matches = context[
            np.isclose(context["Pbar_kw_per_server"], float(start_pbar))
            & np.isclose(context["R_kw_per_server"], float(start_r))
        ]
        if len(matches) != 1:
            raise ValueError("The requested starting P/R must match exactly one sampled pilot row.")
        start = matches.iloc[0]
    else:
        raise ValueError("Provide both --start-pbar and --start-r, or neither.")

    return {
        "all_rows": context_all_weights,
        "start": start,
        "jobs": jobs,
        "initial_weights": initial_weights,
        "weight_columns": weight_columns,
        "p_denominator_watts": float(first["Pbar_denominator_watts"]),
        "r_denominator_watts": float(first["R_denominator_watts"]),
        "p_lower": float(first["Pbar_lower_bound_kw_per_server"]),
        "p_upper": float(first["Pbar_upper_bound_kw_per_server"]),
        "pr_upper": float(first["PR_upper_bound_kw_per_server"]),
        "r_lower": float(first["R_lower_bound_kw_per_server"]),
        "weight_lower": float(first["Weight_Final_Lower_Bound"]),
        "weight_upper": float(first["Weight_Final_Upper_Bound"]),
    }


def model_pr_descent(p_ratio_init,
                     r_ratio_init,
                     workload_mix,
                     model,
                     workload_norm_weights,
                     p_denominator_watts,
                     r_denominator_watts,
                     client_count=1000,
                     util=0.60,
                     p_lower=0.0,
                     p_upper=1.0,
                     pr_upper=1.0,
                     r_lower=0.0,
                     cost_weights=DEFAULT_COST_WEIGHTS,
                     iterations=150,
                     lr=1e-2,
                     device=None):
    """Minimal FlexDC-input adaptation of released CONDOR model_pr_descent()."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    workload_mix = np.asarray(workload_mix, dtype=np.float32)
    workload_size = len(workload_mix)
    if workload_mix.ndim != 2 or workload_mix.shape[1] != 7:
        raise ValueError("workload_mix must have shape J x 7.")

    fixed_jobs = torch.tensor(
        workload_mix[:, :6] / workload_norm_weights[:6],
        dtype=torch.float32,
        device=device,
    )
    workload_weights = torch.tensor(
        workload_mix[:, 6],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    p_ratio = torch.tensor(float(p_ratio_init), dtype=torch.float32, device=device, requires_grad=True)
    r_ratio = torch.tensor(float(r_ratio_init), dtype=torch.float32, device=device, requires_grad=True)

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    softmax = Softmax(dim=0)
    records = []

    for iteration in range(iterations + 1):
        workload = torch.cat([fixed_jobs, workload_weights.unsqueeze(1)], dim=1).unsqueeze(0)
        sim_config = torch.stack([
            p_ratio,
            r_ratio,
            torch.tensor(float(client_count), dtype=torch.float32, device=device),
            torch.tensor(float(util), dtype=torch.float32, device=device),
            torch.tensor(float(workload_size), dtype=torch.float32, device=device),
        ])

        prediction = model(sim_config, workload).reshape(-1)
        objective = (
            float(cost_weights[0]) * prediction[0]
            + float(cost_weights[1]) * prediction[1]
            + float(cost_weights[2]) * prediction[2]
        )

        pbar_kw_per_server = float(
            p_ratio.detach().cpu() * p_denominator_watts / (1000.0 * client_count)
        )
        r_kw_per_server = float(
            r_ratio.detach().cpu() * r_denominator_watts / (1000.0 * client_count)
        )
        predicted_scaled = prediction.detach().cpu().numpy()
        predicted_raw = inverse_scale_predictions(predicted_scaled, client_count, workload_size)

        records.append({
            "Iteration": iteration,
            "Pbar_kw_per_server": pbar_kw_per_server,
            "R_kw_per_server": r_kw_per_server,
            "Pbar_ratio": float(p_ratio.detach().cpu()),
            "R_ratio": float(r_ratio.detach().cpu()),
            **{f"Weight_{i}": float(workload_weights[i].detach().cpu()) for i in range(workload_size)},
            "Predicted_Scaled_cost_power": float(predicted_scaled[0]),
            "Predicted_Scaled_cost_error": float(predicted_scaled[1]),
            "Predicted_Scaled_cost_qos": float(predicted_scaled[2]),
            "Predicted_Raw_cost_power": float(predicted_raw[0]),
            "Predicted_Raw_cost_error": float(predicted_raw[1]),
            "Predicted_Raw_cost_qos": float(predicted_raw[2]),
            "Optimization_Objective": float(objective.detach().cpu()),
        })

        if iteration == iterations:
            break

        objective.backward()
        with torch.no_grad():
            p_ratio -= lr * p_ratio.grad
            r_ratio -= lr * r_ratio.grad

            # Original CONDOR Algorithm 1 / released-code workload-weight update.
            weight_update = softmax(workload_weights - workload_weights.grad) - workload_weights
            workload_weights += lr * weight_update

            # Keep P/R inside the physical FlexDC pilot domain used for training.
            pbar = float(p_ratio.cpu()) * p_denominator_watts / (1000.0 * client_count)
            reserve = float(r_ratio.cpu()) * r_denominator_watts / (1000.0 * client_count)
            pbar = float(np.clip(pbar, p_lower, p_upper))
            reserve_upper = min(pbar - 1e-6, pr_upper - pbar)
            if reserve_upper < r_lower:
                raise ValueError("No valid reserve remains after projecting Pbar.")
            reserve = float(np.clip(reserve, r_lower, reserve_upper))
            p_ratio.copy_(torch.tensor(
                pbar * 1000.0 * client_count / p_denominator_watts,
                dtype=torch.float32,
                device=device,
            ))
            r_ratio.copy_(torch.tensor(
                reserve * 1000.0 * client_count / r_denominator_watts,
                dtype=torch.float32,
                device=device,
            ))

        p_ratio.grad = None
        r_ratio.grad = None
        workload_weights.grad = None

    return pd.DataFrame(records)


def add_actual_label_columns(data):
    rows = []
    for _, row in data.iterrows():
        raw, scaled = condor_labels_from_row(row)
        rows.append({
            "Actual_Raw_cost_power": raw[0],
            "Actual_Raw_cost_error": raw[1],
            "Actual_Raw_cost_qos": raw[2],
            "Actual_Scaled_cost_power": scaled[0],
            "Actual_Scaled_cost_error": scaled[1],
            "Actual_Scaled_cost_qos": scaled[2],
        })
    return pd.DataFrame(rows, index=data.index)


def predict_one(model, row, jobs, weights, workload_norm_weights, device):
    workload = torch.tensor(
        np.column_stack([jobs, weights]) / workload_norm_weights,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    sim_config = torch.tensor([
        float(row["Pbar_ratio"]),
        float(row["R_ratio"]),
        float(row["server_count"]),
        float(row["utilization"]),
        float(row["workload_mix_size"]),
    ], dtype=torch.float32, device=device)
    with torch.no_grad():
        scaled = model(sim_config, workload).detach().cpu().numpy().reshape(-1)
    raw = inverse_scale_predictions(scaled, float(row["server_count"]), float(row["workload_mix_size"]))
    return scaled, raw


def parse_args():
    parser = argparse.ArgumentParser(description="CONDOR inference with FlexDC-generated inputs.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--out-dir", default="condor_flexdc_inference_results")
    parser.add_argument("--workload", default="W1-train")
    parser.add_argument("--server-count", type=int, default=1000)
    parser.add_argument("--utilization", type=float, default=0.60)
    parser.add_argument("--weight-sample-id", type=int, default=0)
    parser.add_argument("--start-pbar", type=float, default=None)
    parser.add_argument("--start-r", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--power-weight", type=float, default=DEFAULT_COST_WEIGHTS[0])
    parser.add_argument("--error-weight", type=float, default=DEFAULT_COST_WEIGHTS[1])
    parser.add_argument("--qos-weight", type=float, default=DEFAULT_COST_WEIGHTS[2])
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    context = load_context(
        args.results_csv,
        args.diagnostics_csv,
        args.workload,
        args.server_count,
        args.utilization,
        args.weight_sample_id,
        args.start_pbar,
        args.start_r,
    )

    # Recompute exactly the workload normalization used by the training loader.
    training_dataset = DCDataset(data_file_path=args.results_csv)
    workload_norm_weights = training_dataset.workload_norm_weights.astype(np.float32)

    model = DataCenterModel()
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.to(device)
    model.eval()

    start = context["start"]
    start_workload = np.column_stack([context["jobs"], context["initial_weights"]])
    cost_weights = [args.power_weight, args.error_weight, args.qos_weight]

    trajectory = model_pr_descent(
        float(start["Pbar_ratio"]),
        float(start["R_ratio"]),
        start_workload,
        model,
        workload_norm_weights,
        context["p_denominator_watts"],
        context["r_denominator_watts"],
        client_count=args.server_count,
        util=args.utilization,
        p_lower=context["p_lower"],
        p_upper=context["p_upper"],
        pr_upper=context["pr_upper"],
        r_lower=context["r_lower"],
        cost_weights=cost_weights,
        iterations=args.iterations,
        lr=args.lr,
        device=device,
    )
    trajectory.to_csv(out_dir / "optimization_trajectory.csv", index=False)

    final = trajectory.iloc[-1]

    # Starting point: actual FlexDC-derived CONDOR labels and model predictions.
    start_raw, start_scaled = condor_labels_from_row(start)
    start_pred_scaled, start_pred_raw = predict_one(
        model,
        start,
        context["jobs"],
        context["initial_weights"],
        workload_norm_weights,
        device,
    )

    # Best sampled pilot row under the same released-example weighted objective.
    sampled = context["all_rows"].copy()
    actual_columns = add_actual_label_columns(sampled)
    sampled = pd.concat([sampled, actual_columns], axis=1)
    sampled["Actual_Optimization_Objective"] = (
        cost_weights[0] * sampled["Actual_Scaled_cost_power"]
        + cost_weights[1] * sampled["Actual_Scaled_cost_error"]
        + cost_weights[2] * sampled["Actual_Scaled_cost_qos"]
    )
    best_sampled = sampled.loc[sampled["Actual_Optimization_Objective"].idxmin()]

    comparison = pd.DataFrame([
        {
            "Configuration": "Starting sampled point",
            "Pbar_kw_per_server": float(start["Pbar_kw_per_server"]),
            "R_kw_per_server": float(start["R_kw_per_server"]),
            "Weights": json.dumps(context["initial_weights"].astype(float).tolist()),
            "Predicted_Scaled_cost_power": start_pred_scaled[0],
            "Predicted_Scaled_cost_error": start_pred_scaled[1],
            "Predicted_Scaled_cost_qos": start_pred_scaled[2],
            "Predicted_Optimization_Objective": float(np.dot(cost_weights, start_pred_scaled)),
            "Actual_Scaled_cost_power": start_scaled[0],
            "Actual_Scaled_cost_error": start_scaled[1],
            "Actual_Scaled_cost_qos": start_scaled[2],
            "Actual_Optimization_Objective": float(np.dot(cost_weights, start_scaled)),
        },
        {
            "Configuration": "CONDOR-selected continuous point",
            "Pbar_kw_per_server": float(final["Pbar_kw_per_server"]),
            "R_kw_per_server": float(final["R_kw_per_server"]),
            "Weights": json.dumps([float(final[f"Weight_{i}"]) for i in range(len(context["initial_weights"]))]),
            "Predicted_Scaled_cost_power": float(final["Predicted_Scaled_cost_power"]),
            "Predicted_Scaled_cost_error": float(final["Predicted_Scaled_cost_error"]),
            "Predicted_Scaled_cost_qos": float(final["Predicted_Scaled_cost_qos"]),
            "Predicted_Optimization_Objective": float(final["Optimization_Objective"]),
            "Actual_Scaled_cost_power": np.nan,
            "Actual_Scaled_cost_error": np.nan,
            "Actual_Scaled_cost_qos": np.nan,
            "Actual_Optimization_Objective": np.nan,
        },
        {
            "Configuration": "Best sampled pilot point",
            "Pbar_kw_per_server": float(best_sampled["Pbar_kw_per_server"]),
            "R_kw_per_server": float(best_sampled["R_kw_per_server"]),
            "Weights": json.dumps(best_sampled[context["weight_columns"]].to_numpy(dtype=float).tolist()),
            "Predicted_Scaled_cost_power": np.nan,
            "Predicted_Scaled_cost_error": np.nan,
            "Predicted_Scaled_cost_qos": np.nan,
            "Predicted_Optimization_Objective": np.nan,
            "Actual_Scaled_cost_power": float(best_sampled["Actual_Scaled_cost_power"]),
            "Actual_Scaled_cost_error": float(best_sampled["Actual_Scaled_cost_error"]),
            "Actual_Scaled_cost_qos": float(best_sampled["Actual_Scaled_cost_qos"]),
            "Actual_Optimization_Objective": float(best_sampled["Actual_Optimization_Objective"]),
        },
    ])
    comparison.to_csv(out_dir / "optimization_comparison_before_validation.csv", index=False)

    candidate = {
        "workload": args.workload,
        "server_count": args.server_count,
        "utilization": args.utilization,
        "starting_weight_sample_id": args.weight_sample_id,
        "starting_pbar_kw_per_server": float(start["Pbar_kw_per_server"]),
        "starting_r_kw_per_server": float(start["R_kw_per_server"]),
        "starting_weights": context["initial_weights"].astype(float).tolist(),
        "optimized_pbar_kw_per_server": float(final["Pbar_kw_per_server"]),
        "optimized_r_kw_per_server": float(final["R_kw_per_server"]),
        "optimized_weights": [float(final[f"Weight_{i}"]) for i in range(len(context["initial_weights"]))],
        "predicted_scaled_cost_power": float(final["Predicted_Scaled_cost_power"]),
        "predicted_scaled_cost_error": float(final["Predicted_Scaled_cost_error"]),
        "predicted_scaled_cost_qos": float(final["Predicted_Scaled_cost_qos"]),
        "predicted_optimization_objective": float(final["Optimization_Objective"]),
        "cost_weights": cost_weights,
        "iterations": args.iterations,
        "learning_rate": args.lr,
        "pbar_bounds_kw_per_server": [context["p_lower"], context["p_upper"]],
        "pr_upper_bound_kw_per_server": context["pr_upper"],
        "r_lower_bound_kw_per_server": context["r_lower"],
        "weight_bounds_from_training_data": [context["weight_lower"], context["weight_upper"]],
        "workload_norm_weights": workload_norm_weights.astype(float).tolist(),
    }
    with (out_dir / "optimized_candidate.json").open("w") as file:
        json.dump(candidate, file, indent=2)

    weights = np.asarray(candidate["optimized_weights"])
    if weights.min() < context["weight_lower"] - 1e-6 or weights.max() > context["weight_upper"] + 1e-6:
        raise ValueError(
            "The optimized weights left the FlexDC pilot training bounds; do not validate this candidate."
        )

    print("Device:", device)
    print("Cost weights [power, error, QoS]:", cost_weights)
    print("Starting P/R:", candidate["starting_pbar_kw_per_server"], candidate["starting_r_kw_per_server"])
    print("Optimized P/R:", candidate["optimized_pbar_kw_per_server"], candidate["optimized_r_kw_per_server"])
    print("Optimized weights:", candidate["optimized_weights"])
    print("Predicted weighted objective:", candidate["predicted_optimization_objective"])
    print("\nSaved:")
    print(out_dir / "optimization_trajectory.csv")
    print(out_dir / "optimization_comparison_before_validation.csv")
    print(out_dir / "optimized_candidate.json")


if __name__ == "__main__":
    main()
