"""
Model-generated FlexDC objective surfaces, analogous to CONDOR Figure 4.

For each selected workload, this script fixes server count, utilization, and
one workload-weight vector, then samples the trained model on a feasible
Pbar/R grid. The plotted height is the model prediction:

    C_total_hat = M_RSR_hat + C_track_hat + C_Qos_hat

The original combined results CSV is used only to recover the exact workload
profile and weights. Costs come from the trained model, not the simulator.
"""

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd
import torch

from data_center_model import DataCenterModel


WORKLOAD_NORMALIZER = np.array([1000, 1000, 3600, 3600, 1, 1, 1], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot trained FlexDC model cost surfaces.")
    parser.add_argument("--results-csv", required=True, help="Combined FlexDC results CSV.")
    parser.add_argument("--model-file", required=True, help="Saved am_flexdc model state_dict file.")
    parser.add_argument("--out-dir", default="flexdc_model_surfaces")
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["W1-train", "W2-short-qos2-2.5-2.5-3"],
        help="Workload_Name values to plot.",
    )
    parser.add_argument("--server-count", type=int, default=1000)
    parser.add_argument("--utilization", type=float, default=0.60)
    parser.add_argument(
        "--weight-sample-id",
        type=int,
        default=0,
        help="Use 0 for the equal-weight vector in this pilot.",
    )
    parser.add_argument("--grid-points", type=int, default=12)
    parser.add_argument("--pr-upper-factor", type=float, default=1.2)
    parser.add_argument("--r-lower", type=float, default=0.01)
    
    parser.add_argument(
        "--show-3d",
        action="store_true",
        help="Open the saved 3D figure for interactive rotation.",
    )
    
    return parser.parse_args()


def load_context(results, workload_name, server_count, utilization, weight_sample_id):
    selected = results[
        (results["Workload_Name"] == workload_name)
        & (results["server_count"].astype(int) == server_count)
        & np.isclose(results["utilization"].astype(float), utilization)
        & (results["Weight_Sample_ID"].astype(int) == weight_sample_id)
    ].copy()

    if selected.empty:
        raise ValueError(f"No rows found for {workload_name}, N={server_count}, U={utilization}, weight={weight_sample_id}.")

    if selected["workload_mix"].nunique() != 1:
        raise ValueError(f"{workload_name} has more than one workload profile in the selected context.")

    weight_cols = sorted(
        [column for column in selected.columns if column.startswith("Weight_") and column != "Weight_Sample_ID"],
        key=lambda column: int(column.split("_")[-1]),
    )
    first = selected.iloc[0]
    jobs = np.asarray(ast.literal_eval(first["workload_mix"]), dtype=np.float32)
    weights = first[weight_cols].to_numpy(dtype=np.float32).reshape(-1, 1)

    if jobs.ndim != 2 or jobs.shape[1] != 6 or jobs.shape[0] != weights.shape[0]:
        raise ValueError("Expected a six-feature job matrix and one corresponding weight per job type.")

    workload_mix = np.column_stack([jobs, weights])
    return {
        "workload_mix": workload_mix,
        "workload_mix_size": int(first["workload_mix_size"]),
        "pbar_min": float(selected["Pbar_kw_per_server"].min()),
        "pbar_max": float(selected["Pbar_kw_per_server"].max()),
        "weights": weights.ravel(),
    }


def make_feasible_grid(context, grid_points, r_lower, pr_upper_factor):
    pbar_values = np.linspace(context["pbar_min"], context["pbar_max"], grid_points)
    rows = []

    for pbar in pbar_values:
        r_max = min(pbar, pr_upper_factor * context["pbar_max"] - pbar)
        if r_max < r_lower:
            raise ValueError(f"No feasible R values at Pbar={pbar:.6f}: R_max={r_max:.6f} < R_lower={r_lower:.6f}.")
        for reserve in np.linspace(r_lower, r_max, grid_points):
            rows.append((pbar, reserve))

    return np.asarray(rows, dtype=np.float32)


def predict_surface(model, context, server_count, utilization, grid):
    features = np.column_stack([
        grid[:, 0],
        grid[:, 1],
        np.full(len(grid), server_count, dtype=np.float32),
        np.full(len(grid), utilization, dtype=np.float32),
        np.full(len(grid), context["workload_mix_size"], dtype=np.float32),
    ]).astype(np.float32)

    workload = np.repeat(context["workload_mix"][None, :, :], len(grid), axis=0)
    workload = workload / WORKLOAD_NORMALIZER

    with torch.no_grad():
        prediction = model(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(workload, dtype=torch.float32),
        ).cpu().numpy()

    return prediction


def plot_surface(ax, workload_name, grid, total_cost):
    x = grid[:, 0]
    y = grid[:, 1]
    tri = Triangulation(x, y)

    surface = ax.plot_trisurf(
        tri,
        total_cost,
        cmap="viridis",
        linewidth=0.25,
        edgecolor=(0.15, 0.15, 0.15, 0.30),
        antialiased=True,
    )

    ax.set_title(f"Predicted Cost Landscape: {workload_name}", fontsize=16, pad=18)
    ax.set_xlabel(r"Baseline Power $\bar{P}$ (kW/server)", labelpad=10)
    ax.set_ylabel(r"Reserve $R$ (kW/server)", labelpad=10)
    ax.set_zlabel(r"Predicted $\hat{C}_{total}$", labelpad=9)
    ax.set_box_aspect((1.15, 1.0, 0.76))
    ax.view_init(elev=28, azim=-132)
    return surface


def main():
    args = parse_args()
    if args.grid_points < 2:
        raise ValueError("--grid-points must be at least 2.")

    results = pd.read_csv(args.results_csv)
    required = {
        "Workload_Name", "Weight_Sample_ID", "Pbar_kw_per_server", "R_kw_per_server",
        "server_count", "utilization", "workload_mix_size", "workload_mix",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Results CSV is missing: {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = DataCenterModel().to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()

    fig = plt.figure(figsize=(8.2 * len(args.workloads), 7.2))
    predictions = []

    for index, workload_name in enumerate(args.workloads, start=1):
        context = load_context(
            results,
            workload_name,
            args.server_count,
            args.utilization,
            args.weight_sample_id,
        )
        grid = make_feasible_grid(context, args.grid_points, args.r_lower, args.pr_upper_factor)
        components = predict_surface(model, context, args.server_count, args.utilization, grid)
        total_cost = components.sum(axis=1)

        ax = fig.add_subplot(1, len(args.workloads), index, projection="3d")
        surface = plot_surface(ax, workload_name, grid, total_cost)
        colorbar = fig.colorbar(surface, ax=ax, shrink=0.64, pad=0.10)
        colorbar.set_label(r"Predicted $\hat{C}_{total}$", labelpad=8)

        frame = pd.DataFrame({
            "Workload_Name": workload_name,
            "server_count": args.server_count,
            "utilization": args.utilization,
            "Weight_Sample_ID": args.weight_sample_id,
            "Pbar_kw_per_server": grid[:, 0],
            "R_kw_per_server": grid[:, 1],
            "Predicted_M_RSR": components[:, 0],
            "Predicted_C_track": components[:, 1],
            "Predicted_C_Qos": components[:, 2],
            "Predicted_C_total": total_cost,
        })
        predictions.append(frame)

        print(
            f"{workload_name}: N={args.server_count}, U={args.utilization:.2f}, "
            f"weights={np.round(context['weights'], 4).tolist()}, points={len(frame)}"
        )

    fig.suptitle(
        "Trained FlexDC-Objective CONDOR Surface Approximation\n"
        f"Fixed N={args.server_count}, U={args.utilization:.2f}, Weight_Sample_ID={args.weight_sample_id}",
        fontsize=18,
        y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.96, top=0.85, bottom=0.06, wspace=0.22)

    figure_path = out_dir / "flexdc_model_full_objective_surfaces.png"
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    if args.show_3d:
        plt.show()

    plt.close(fig)
    prediction_path = out_dir / "flexdc_model_surface_predictions.csv"
    pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False)

    print(f"Saved: {figure_path}")
    print(f"Saved: {prediction_path}")


if __name__ == "__main__":
    main()
