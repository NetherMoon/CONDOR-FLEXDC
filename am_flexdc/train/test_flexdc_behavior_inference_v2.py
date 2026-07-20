#!/usr/bin/env python3
"""Structural and integration tests for CONDOR-FlexDC behavior-v2 inference."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from am_flexdc_behavior_inference_utilities_v2 import (
    OptimizationSettings,
    build_differentiable_features,
    build_flexdc_wizard_command,
    calculate_pr_bounds,
    calculate_weight_bounds,
    load_behavior_model,
    make_feature_row,
    optimize_candidates,
    parameterize_weights,
    predict_configuration,
    read_experiment_config,
    read_workload_config,
    reconstruct_costs_numpy,
    resolve_safety_limits,
)
from am_flexdc_behavior_training_utilities_v2 import parse_list, parse_workload_mix


def assert_close(a, b, tolerance=1e-6, message=""):
    if not np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True):
        raise AssertionError(message or f"Values differ: {a} vs {b}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--diagnostics-csv", default=None)
    parser.add_argument("--flexdc-root", default=None)
    parser.add_argument("--gradient-config", default=None)
    parser.add_argument("--cluster-config", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-optimizer-smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("T0: Load checkpoint and configs")
    loaded = load_behavior_model(args.checkpoint, device_name=args.device)
    workload = read_workload_config(args.workload_config)
    experiment = read_experiment_config(args.experiment_config)
    bounds = calculate_pr_bounds(workload)
    safety = resolve_safety_limits(loaded.constants, tracking_margin=0.04, qos_margin=0.01)
    assert loaded.model.config.dim_job_mix == 13
    assert loaded.model.config.dim_dc_features == 12
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    assert safety.selection_tracking_limit == 0.26
    assert abs(safety.selection_qos_limit - 0.09) < 1e-12
    print("PASS")

    print("T1: Predict one configuration twice and verify determinism")
    pbar = min(max(0.472128, bounds.pbar_lower_kw_per_server + 0.001), bounds.pbar_upper_kw_per_server - 0.001)
    reserve_max = min(pbar, bounds.pr_upper_kw_per_server - pbar)
    reserve = min(max(0.102206, bounds.r_lower_kw_per_server + 0.001), reserve_max - 0.001)
    weights = np.full(workload.job_count, 1.0 / workload.job_count)
    first, first_jobs = predict_configuration(
        loaded,
        workload=workload,
        experiment=experiment,
        pbar_kw_per_server=pbar,
        r_kw_per_server=reserve,
        weights=weights,
        safety=safety,
        bounds=bounds,
    )
    second, second_jobs = predict_configuration(
        loaded,
        workload=workload,
        experiment=experiment,
        pbar_kw_per_server=pbar,
        r_kw_per_server=reserve,
        weights=weights,
        safety=safety,
        bounds=bounds,
    )
    assert_close(first["Predicted_P90_Tracking"], second["Predicted_P90_Tracking"], 1e-10)
    assert_close(first_jobs["Predicted_Pj"], second_jobs["Predicted_Pj"], 1e-10)
    assert np.all((first_jobs["Predicted_Pj"] >= 0) & (first_jobs["Predicted_Pj"] <= 1))
    print("PASS")

    print("T2: Exact feature parity with training feature builders")
    _, tokens, global_features, _ = make_feature_row(
        pbar_kw_per_server=pbar,
        r_kw_per_server=reserve,
        weights=weights,
        workload=workload,
        experiment=experiment,
    )
    # The checkpoint may be loaded on CUDA when --device auto is used.
    # Create the differentiable inputs on the same device as the model; otherwise
    # PyTorch raises a CPU/CUDA matrix-multiplication device mismatch.
    device = loaded.device
    p_t = torch.tensor(
        [pbar], dtype=torch.float32, device=device, requires_grad=True
    )
    r_t = torch.tensor(
        [reserve], dtype=torch.float32, device=device, requires_grad=True
    )
    w_t = torch.tensor(
        weights[None, :], dtype=torch.float32, device=device, requires_grad=True
    )
    global_t, tokens_t, mask_t = build_differentiable_features(
        pbar=p_t,
        reserve=r_t,
        weights=w_t,
        workload=workload,
        experiment=experiment,
        metadata=loaded.metadata,
    )
    token_expected = (
        tokens - np.asarray(loaded.metadata.token_feature_mean)
    ) / np.asarray(loaded.metadata.token_feature_std)
    global_expected = (
        global_features - np.asarray(loaded.metadata.global_feature_mean)
    ) / np.asarray(loaded.metadata.global_feature_std)
    assert_close(tokens_t.detach().cpu().numpy()[0], token_expected, 2e-5)
    assert_close(global_t.detach().cpu().numpy()[0], global_expected, 2e-5)
    assert bool(mask_t.all().item())
    output = loaded.model(global_t, tokens_t, mask_t)
    objective = (
        output["tracking_logs"].sum()
        + output["qos_probabilities"].sum()
    )
    objective.backward()
    assert p_t.grad is not None and torch.isfinite(p_t.grad).all()
    assert r_t.grad is not None and torch.isfinite(r_t.grad).all()
    assert w_t.grad is not None and torch.isfinite(w_t.grad).all()
    print("PASS")

    print("T3: Analytical cost lineage against a real training row")
    if args.results_csv:
        data = pd.read_csv(args.results_csv, nrows=2000)
        matching = data[data["Workload_Config"].astype(str).str.contains(Path(args.workload_config).name, regex=False)]
        row = (matching if len(matching) else data).iloc[0]
        probs = parse_list(row["QoS_Delay_Probabilities"], "QoS_Delay_Probabilities")
        exp_for_row = read_experiment_config(
            args.experiment_config,
            server_count_override=int(row["server_count"]),
            utilization_override=float(row["utilization"]),
        )
        costs = reconstruct_costs_numpy(
            pbar_kw_per_server=float(row["Pbar_kw_per_server"]),
            r_kw_per_server=float(row["R_kw_per_server"]),
            mean_tracking=float(row["Mtrack_Error_MeanAbs_Normalized"]),
            p90_tracking=float(row["Ctrack_Epsilon_90th"]),
            qos_probabilities=probs,
            experiment=exp_for_row,
            constants=loaded.constants,
        )
        assert_close(costs["Simulator_Power_Cost"], float(row["Simulator_Power_Cost"]), 1e-5)
        assert_close(costs["Predicted_Mtrack_Cost"], float(row["Mtrack_Cost"]), 1e-5)
        assert_close(costs["Predicted_M_RSR"], float(row["Simulator_RSR_Total_Cost"]), 1e-5)
        assert_close(costs["Predicted_Full_Objective"], float(row["Diagnostic_FullPaperObjective_Cost"]), 1e-5)
        print("PASS")
    else:
        print("SKIP: --results-csv not supplied")

    print("T4: Optimization parameterization/top-k contract")
    automatic_weight_bounds = calculate_weight_bounds(workload.job_count, experiment.server_count)
    weight_logits_test = torch.randn(8, workload.job_count, device=loaded.device, requires_grad=True)
    bounded_weights_test = parameterize_weights(
        weight_logits_test,
        automatic_weight_bounds.final_lower,
        automatic_weight_bounds.final_upper,
    )
    assert torch.allclose(
        bounded_weights_test.sum(dim=1),
        torch.ones(8, device=loaded.device),
        atol=2e-6,
        rtol=0,
    )
    assert float(bounded_weights_test.min().detach().cpu()) >= automatic_weight_bounds.final_lower - 2e-6
    assert float(bounded_weights_test.max().detach().cpu()) <= automatic_weight_bounds.final_upper + 2e-6
    bounded_weights_test.square().sum().backward()
    assert weight_logits_test.grad is not None and torch.isfinite(weight_logits_test.grad).all()

    if args.run_optimizer_smoke:
        settings = OptimizationSettings(
            starts=2,
            iterations=1,
            learning_rate=0.02,
            minimum_learning_rate=0.002,
            mode="margin_constrained",
            top_k=2,
            random_seed=7,
            log_every=1,
        )
        candidates, top_k, trajectory = optimize_candidates(
            loaded,
            workload=workload,
            experiment=experiment,
            bounds=bounds,
            safety=safety,
            settings=settings,
            initial_pbar=pbar,
            initial_reserve=reserve,
            initial_weights=weights,
        )
        assert len(candidates) == 2
        assert len(trajectory) >= 1
        assert np.all(candidates["R_kw_per_server"] <= candidates["Pbar_kw_per_server"] + 1e-7)
        assert np.all(candidates["Pbar_plus_R"] <= bounds.pr_upper_kw_per_server + 1e-6)
        for value in candidates["weights"]:
            assert_close(sum(value), 1.0, 2e-6)
            assert min(value) >= automatic_weight_bounds.final_lower - 2e-6
            assert max(value) <= automatic_weight_bounds.final_upper + 2e-6
        assert candidates["Weight_Bounds_Pass"].all()
        if len(top_k) > 1:
            assert top_k["Candidate_Rank"].tolist() == list(range(1, len(top_k) + 1))
        print("PASS")
    else:
        # T2 already verified finite gradients through Pbar, R, and weights.
        # The full batched-optimizer smoke is intentionally run on GPU in the
        # Colab notebook because this 3.9M-parameter model is very slow on CPU.
        print("SKIP: pass --run-optimizer-smoke on a CUDA runtime")

    print("T5: FlexDC wizard command dry-run contract")
    if args.flexdc_root and args.gradient_config and args.cluster_config:
        command, cwd, env = build_flexdc_wizard_command(
            python_executable=sys.executable,
            flexdc_root=args.flexdc_root,
            gradient_config=args.gradient_config,
            experiment_config=args.experiment_config,
            cluster_config=args.cluster_config,
            workload_config=args.workload_config,
            output_label="behavior_v2_test",
            pbar_kw_per_server=pbar,
            r_kw_per_server=reserve,
            weights=weights,
            utilization=experiment.utilization,
        )
        assert Path(cwd).name == "peacsim"
        assert "--auto-workload-pr-sweep" in command
        assert "--weight-vectors" in command
        assert str(Path(args.flexdc_root).resolve() / "src") in env["PYTHONPATH"]
        print("PASS")
    else:
        print("SKIP: FlexDC path/config arguments not supplied")

    print("\nALL INFERENCE V2 TESTS PASSED")


if __name__ == "__main__":
    main()
