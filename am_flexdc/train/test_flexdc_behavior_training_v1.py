"""Structural smoke tests for FlexDC behavior training files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_center_model_flexdc_behavior import DataCenterBehaviorModel, FlexDCBehaviorModelConfig
from am_flexdc_behavior_training_utilities import behavior_loss, collate_behavior_batch


def make_item(j: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    workload = torch.rand((j, 7), generator=generator) + 0.1
    qos = torch.rand((j,), generator=generator)
    return {
        "features": torch.rand((5,), generator=generator),
        "workload": workload,
        "tracking_logs": torch.rand((2,), generator=generator),
        "qos_probabilities": qos,
        "raw_mean_tracking": torch.rand((), generator=generator),
        "raw_p90_tracking": torch.rand((), generator=generator),
        "simulator_power_cost": torch.rand((), generator=generator),
        "actual_m_rsr": torch.rand((), generator=generator),
        "actual_objective": torch.rand((), generator=generator),
        "r_actual_watts": torch.rand((), generator=generator) * 1000,
        "tracking_price_coefficient": torch.tensor(2.77777777777778e-8),
        "hour_seconds": torch.tensor(3600.0),
        "plan_row_id": f"synthetic-{j}-{seed}",
    }


def main() -> None:
    torch.manual_seed(7)
    config = FlexDCBehaviorModelConfig(
        st_dim_hidden=64,
        st_dim_output=123,
        linear_dim_hidden=64,
        qos_projection_dim=32,
    )
    model = DataCenterBehaviorModel(config).eval()

    # Mixed variable-J batching and mask creation.
    batch = collate_behavior_batch([make_item(2, 1), make_item(4, 2), make_item(6, 3)])
    assert batch["workload"].shape == (3, 6, 7)
    assert batch["mask"].sum(dim=1).tolist() == [2, 4, 6]
    output = model(batch["features"], batch["workload"], batch["mask"])
    assert output["tracking_logs"].shape == (3, 2)
    assert output["qos_probabilities"].shape == (3, 6)
    assert torch.all(output["qos_probabilities"][0, 2:] == 0)
    assert torch.all(output["qos_probabilities"][1, 4:] == 0)

    # A sample evaluated alone must equal itself inside a padded mixed-J batch.
    item = make_item(3, 10)
    alone = collate_behavior_batch([item])
    mixed = collate_behavior_batch([item, make_item(7, 11)])
    with torch.no_grad():
        out_alone = model(alone["features"], alone["workload"], alone["mask"])
        out_mixed = model(mixed["features"], mixed["workload"], mixed["mask"])
    torch.testing.assert_close(out_alone["tracking_logs"][0], out_mixed["tracking_logs"][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_alone["qos_probabilities"][0, :3], out_mixed["qos_probabilities"][0, :3], rtol=1e-5, atol=1e-6)

    # Permuting job order should preserve scalar tracking outputs and permute P_j outputs.
    permutation = torch.tensor([2, 0, 1])
    perm_item = dict(item)
    perm_item["workload"] = item["workload"][permutation]
    perm_item["qos_probabilities"] = item["qos_probabilities"][permutation]
    perm_batch = collate_behavior_batch([perm_item])
    with torch.no_grad():
        out_perm = model(perm_batch["features"], perm_batch["workload"], perm_batch["mask"])
    torch.testing.assert_close(out_alone["tracking_logs"][0], out_perm["tracking_logs"][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        out_alone["qos_probabilities"][0, permutation],
        out_perm["qos_probabilities"][0, :3],
        rtol=1e-5,
        atol=1e-6,
    )

    # Padded positions must not receive gradients through the QoS loss.
    model.train()
    mixed = collate_behavior_batch([make_item(2, 20), make_item(5, 21)])
    mixed["workload"].requires_grad_(True)
    pred = model(mixed["features"], mixed["workload"], mixed["mask"])
    qos_loss = (((pred["qos_probabilities"] - mixed["qos_probabilities"]) ** 2) * mixed["mask"].float()).sum()
    qos_loss.backward()
    assert torch.all(mixed["workload"].grad[0, 2:] == 0)

    print("ALL FLEXDC BEHAVIOR STRUCTURAL TESTS PASSED")


if __name__ == "__main__":
    main()
