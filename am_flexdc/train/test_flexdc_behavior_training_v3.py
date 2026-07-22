"""Structural and training-utility tests for FlexDC Model V3.

Run with:
    python test_flexdc_behavior_training_v3.py
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_center_model_flexdc_behavior_v3 import (
    DataCenterBehaviorModel,
    FlexDCBehaviorModelConfig,
)
from am_flexdc_behavior_training_utilities_v3 import (
    GLOBAL_FEATURE_NAMES,
    TOKEN_FEATURE_NAMES,
    build_global_features_raw,
    build_token_features_raw,
    build_training_sampling_weights,
    collate_behavior_batch,
    cosine_warmup_lr,
    prepare_behavior_data,
    preassigned_train_validation_test_split,
)


def assert_close(a, b, *, atol=1e-6, message=""):
    if not torch.allclose(a, b, atol=atol, rtol=0):
        raise AssertionError(message or f"Tensors differ; max={torch.max(torch.abs(a-b)).item()}")


def test_feature_dimensions():
    mix = np.asarray(
        [
            [199, 755, 337, 1321, 3, 1],
            [201, 763, 1185, 4592, 3, 1],
            [200, 742, 1242, 4921, 3, 1],
            [202, 726, 1104, 4202, 3, 1],
        ],
        dtype=float,
    )
    weights = np.asarray([0.25, 0.25, 0.25, 0.25])
    token = build_token_features_raw(mix, weights, utilization=0.6, server_count=1000)
    row = pd.Series(
        {
            "Pbar_kw_per_server": 0.48,
            "R_kw_per_server": 0.11,
            "Pbar_ratio": 0.97,
            "R_ratio": 0.35,
            "utilization": 0.6,
            "server_count": 1000,
            "workload_mix_size": 4,
        }
    )
    glob = build_global_features_raw(row)
    assert token.shape == (4, len(TOKEN_FEATURE_NAMES))
    assert glob.shape == (len(GLOBAL_FEATURE_NAMES),)
    assert np.isfinite(token).all() and np.isfinite(glob).all()


def make_variable_batch():
    torch.manual_seed(7)
    items = []
    for j in [2, 4, 6]:
        items.append(
            {
                "features": torch.randn(12),
                "workload": torch.randn(j, 13),
                "tracking_logs": torch.randn(2),
                "qos_probabilities": torch.sigmoid(torch.randn(j)),
                "raw_mean_tracking": torch.rand(()),
                "raw_p90_tracking": torch.rand(()),
                "simulator_power_cost": torch.rand(()),
                "actual_m_rsr": torch.rand(()),
                "actual_objective": torch.rand(()),
                "r_actual_watts": torch.rand(()),
                "tracking_price_coefficient": torch.rand(()),
                "hour_seconds": torch.tensor(3600.0),
                "plan_row_id": f"row-{j}",
                "workload_name": f"W{j}",
                "context_id": f"W{j}|N=1000|U=0.6",
                "server_count": 1000,
                "utilization": 0.6,
            }
        )
    return collate_behavior_batch(items)


def small_config():
    return FlexDCBehaviorModelConfig(st_dim_hidden=32, st_num_heads=4, global_projection_dim=16, linear_dim_hidden=32, qos_projection_dim=16)


def test_variable_j_and_masks():
    config = small_config()
    model = DataCenterBehaviorModel(config).eval()
    batch = make_variable_batch()
    with torch.no_grad():
        out = model(batch["features"], batch["workload"], batch["mask"])
    assert out["tracking_logs"].shape == (3, 2)
    assert out["qos_probabilities"].shape == (3, 6)
    assert torch.count_nonzero(out["qos_probabilities"][0, 2:]) == 0
    assert torch.count_nonzero(out["qos_probabilities"][1, 4:]) == 0


def test_padding_batch_invariance():
    torch.manual_seed(10)
    model = DataCenterBehaviorModel(small_config()).eval()
    feature = torch.randn(1, 12)
    jobs = torch.randn(1, 4, 13)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        alone = model(feature, jobs, mask)
        padded_jobs = torch.cat([jobs, torch.randn(1, 3, 13) * 100], dim=1)
        padded_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        padded = model(feature, padded_jobs, padded_mask)
    assert_close(alone["tracking_logs"], padded["tracking_logs"], atol=1e-5)
    assert_close(alone["qos_probabilities"], padded["qos_probabilities"][:, :4], atol=1e-5)


def test_permutation_equivariance():
    torch.manual_seed(11)
    model = DataCenterBehaviorModel(small_config()).eval()
    feature = torch.randn(1, 12)
    jobs = torch.randn(1, 4, 13)
    mask = torch.ones(1, 4, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        base = model(feature, jobs, mask)
        permuted = model(feature, jobs[:, permutation], mask)
    assert_close(base["tracking_logs"], permuted["tracking_logs"], atol=1e-5)
    assert_close(
        base["qos_probabilities"][:, permutation],
        permuted["qos_probabilities"],
        atol=1e-5,
    )


def test_zero_gradient_to_padding():
    torch.manual_seed(12)
    model = DataCenterBehaviorModel(small_config())
    features = torch.randn(1, 12, requires_grad=True)
    jobs = torch.randn(1, 6, 13, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    out = model(features, jobs, mask)
    loss = out["tracking_logs"].sum() + out["qos_probabilities"].sum()
    loss.backward()
    assert torch.max(torch.abs(jobs.grad[:, 4:])).item() < 1e-8


def test_sampler_and_schedule():
    df = pd.DataFrame(
        {
            "_group_id": [f"g-{index}" for index in range(100)],
            "_context_id": ["a"] * 90 + ["b"] * 10,
            "_actual_both_pass": [False] * 95 + [True] * 5,
            "_tracking_boundary": [False] * 80 + [True] * 20,
            "_qos_boundary": [False] * 85 + [True] * 15,
        }
    )
    weights, audit = build_training_sampling_weights(df, mode="balanced")
    assert weights is not None
    assert np.isclose(weights.sum(), 1.0)
    assert audit["expected_sampled_feasible_share"] > audit["raw_feasible_share"]
    lrs = [
        cosine_warmup_lr(
            epoch,
            max_epochs=20,
            base_lr=3e-4,
            min_lr=1e-6,
            warmup_epochs=5,
        )
        for epoch in range(20)
    ]
    assert lrs[0] < lrs[4]
    assert math.isclose(lrs[4], 3e-4, rel_tol=1e-8)
    assert lrs[-1] <= 1.01e-6



def _synthetic_behavior_row(
    *,
    plan_row_id: str,
    base_id: str,
    split: str,
    seed: int,
    pbar: float,
    reserve: float,
    p90: float,
    qos0: float,
    group_size: int,
):
    mix = [
        [199, 755, 337, 1321, 3, 1],
        [201, 763, 1185, 4592, 3, 1],
        [200, 742, 1242, 4921, 3, 1],
        [202, 726, 1104, 4202, 3, 1],
    ]
    qos = [qos0, 0.0, 0.0, 0.0]
    mtrack = 0.01
    power = 20.0
    m_rsr = power + mtrack
    objective = m_rsr + 0.1 + 48.0
    return {
        "Plan_Row_ID": plan_row_id,
        "Base_Plan_Row_ID": base_id,
        "Data_Split": split,
        "Simulation_Seed": seed,
        "Base_Group_Normalization_Weight": 1.0 / group_size,
        "Workload_Name": "W1-train-qos3333",
        "Pbar_kw_per_server": pbar,
        "R_kw_per_server": reserve,
        "Pbar_ratio": pbar / 0.5,
        "R_ratio": reserve / 0.3,
        "server_count": 1000,
        "utilization": 0.6,
        "workload_mix_size": 4,
        "workload_mix": repr(mix),
        "weights": repr([0.25, 0.25, 0.25, 0.25]),
        "Mtrack_Error_MeanAbs_Normalized": mtrack,
        "Ctrack_Epsilon_90th": p90,
        "QoS_Delay_Probabilities": repr(qos),
        "Simulator_Power_Cost": power,
        "Simulator_RSR_Total_Cost": m_rsr,
        "Diagnostic_FullPaperObjective_Cost": objective,
        "Mtrack_Price_Coefficient_piE": 1.0,
        "Mtrack_Hour_Seconds": 3600.0,
        "R_actual_watts": reserve * 1000 * 1000,
    }


def test_preassigned_split_and_repeat_group_normalization():
    rows = []
    # Three train base groups; the first has three seed realizations.
    for seed in [20, 21, 22]:
        rows.append(
            _synthetic_behavior_row(
                plan_row_id=f"train-repeat-{seed}",
                base_id="train-repeat",
                split="train",
                seed=seed,
                pbar=0.40,
                reserve=0.10,
                p90=0.25 + 0.001 * (seed - 20),
                qos0=0.08 + 0.001 * (seed - 20),
                group_size=3,
            )
        )
    for index in [1, 2]:
        rows.append(
            _synthetic_behavior_row(
                plan_row_id=f"train-{index}",
                base_id=f"train-{index}",
                split="train",
                seed=20,
                pbar=0.41 + 0.01 * index,
                reserve=0.11,
                p90=0.2,
                qos0=0.05,
                group_size=1,
            )
        )
    rows.append(
        _synthetic_behavior_row(
            plan_row_id="validation-1",
            base_id="validation-1",
            split="validation",
            seed=20,
            pbar=0.50,
            reserve=0.12,
            p90=0.3,
            qos0=0.1,
            group_size=1,
        )
    )
    # Give test an extreme Pbar so training-only normalization can be checked.
    rows.append(
        _synthetic_behavior_row(
            plan_row_id="test-1",
            base_id="test-1",
            split="test",
            seed=20,
            pbar=9.0,
            reserve=0.13,
            p90=0.4,
            qos0=0.2,
            group_size=1,
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "sweep_v3_training_ready.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        bundle = prepare_behavior_data(
            results_csv=csv_path,
            diagnostics_csv=None,
            batch_size=4,
            split_mode="preassigned",
            split_column="Data_Split",
            sampler_mode="natural",
            repeat_group_normalization=True,
        )

    assert bundle.metadata.train_group_count == 3
    assert bundle.metadata.validation_group_count == 1
    assert bundle.metadata.test_group_count == 1
    assert bundle.metadata.train_row_count == 5
    assert bundle.metadata.validation_row_count == 1
    assert bundle.metadata.test_row_count == 1
    assert bundle.test_loader is not None
    assert set(bundle.train_dataframe["_group_id"]) == {
        "train-repeat", "train-1", "train-2"
    }
    assert set(bundle.validation_dataframe["_group_id"]) == {"validation-1"}
    assert set(bundle.test_dataframe["_group_id"]) == {"test-1"}

    # The three seed rows together receive the same mass as either one-row group.
    probabilities = pd.Series(bundle.train_sampling_weights)
    group_totals = probabilities.groupby(
        bundle.train_dataframe["_group_id"].reset_index(drop=True)
    ).sum()
    assert np.allclose(group_totals.to_numpy(), np.full(3, 1.0 / 3.0))

    # Global Pbar mean comes only from train, not validation/test.
    pbar_index = GLOBAL_FEATURE_NAMES.index("Pbar_kw_per_server")
    expected_train_pbar_mean = bundle.train_dataframe[
        "Pbar_kw_per_server"
    ].astype(float).mean()
    assert np.isclose(
        bundle.metadata.global_feature_mean[pbar_index],
        expected_train_pbar_mean,
    )


def test_preassigned_split_rejects_seed_leakage():
    df = pd.DataFrame(
        {
            "_group_id": ["same", "same", "v", "t"],
            "_context_id": ["c"] * 4,
            "Data_Split": ["train", "test", "validation", "test"],
        }
    )
    try:
        preassigned_train_validation_test_split(df)
    except AssertionError:
        return
    raise AssertionError("Expected cross-split seed group to be rejected")

def main():
    tests = [
        test_feature_dimensions,
        test_variable_j_and_masks,
        test_padding_batch_invariance,
        test_permutation_equivariance,
        test_zero_gradient_to_padding,
        test_sampler_and_schedule,
        test_preassigned_split_and_repeat_group_normalization,
        test_preassigned_split_rejects_seed_leakage,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("ALL MODEL V3 / SWEEP V3 STRUCTURAL TESTS PASSED")


if __name__ == "__main__":
    main()
