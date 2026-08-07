#!/usr/bin/env python3
"""Static parity guards for the three repaired generic notebooks."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    "training": ROOT / "FlexDC_Generic_Full_Parity_Training_and_Evaluation.ipynb",
    "inference": ROOT / "FlexDC_Generic_Full_Parity_Inference_Validation.ipynb",
    "paired": ROOT / "FlexDC_Generic_Full_Parity_Paired_Comparison.ipynb",
}

REQUIRED_TOKENS = {
    "training": [
        "clone_or_update_repository", "USE_GOOGLE_DRIVE_DATA", "multipart_upload",
        "train_behavior_model", "RECOVERY_MODE", "best_objective", "best_feasibility",
        "final_behavior_test_evaluation", "run_profile_holdout_evaluation",
        "run_scenario_suite", "RUN_ACTUAL_FLEXDC_VALIDATION", "wandb", "package_paths",
    ],
    "inference": [
        "CLONE_CONDOR_REPO", "CLONE_FLEXDC_REPO", "ARTIFACT_MODE",
        "select_checkpoint", "margin_calibration_table", "predict_configuration",
        "optimize_candidates", "RUN_FLEXDC_VALIDATION", "run_flexdc_validation",
        "RUN_FIXED_PROFILE_SUITE", "RUN_LEGACY_CUSTOM_ROUNDS", "RUN_ALL_WORKLOADS",
        "compare_models_on_points", "REFERENCE_POINTS_CSV", "wandb", "package_paths",
    ],
    "paired": [
        "CLONE_CONDOR_REPO", "CLONE_FLEXDC_REPO", "MODEL_ARTIFACT_INPUTS",
        "EXPLICIT_START_PAIRS", "run_paired_cases", "RUN_PREDICT_ONE", "RUN_OPTIMIZE_ONE",
        "RUN_ROUND_1", "RUN_ROUND_2", "RUN_ROUND_3", "RUN_ROUND_4", "RUN_ROUND_5",
        "RUN_ALL_WORKLOAD_PAIRED_OPTIMIZATION", "wandb", "package_paths",
    ],
}


def validate_notebook(label: str, path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4, path
    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{path.name}:cell{index}")
    missing = [token for token in REQUIRED_TOKENS[label] if token not in joined]
    assert not missing, f"{path.name} missing parity tokens: {missing}"


def main() -> None:
    for label, path in NOTEBOOKS.items():
        assert path.exists(), path
        validate_notebook(label, path)
        print(f"PASS: {label}: {path.name}")
    print("ALL FULL-PARITY NOTEBOOK STATIC GUARDS PASSED")


if __name__ == "__main__":
    main()
