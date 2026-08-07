#!/usr/bin/env python3
"""Structural tests for generic inference orchestration and parity helpers."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from flexdc_colab_orchestration import (
    package_paths, reassemble_from_manifest, reassemble_multipart_file, sha256_file,
)
from flexdc_inference_orchestration import (
    ScenarioDefinition,
    aggregate_validation,
    build_profile_suite,
    classify_workload_for_model,
    restore_prior_outputs,
)
from flexdc_presentation import build_report
from flexdc_behavior_inference_utilities import (
    ExperimentSpec, OptimizationSettings, WorkloadSpec,
    calculate_pr_bounds, prepare_ratio_starts,
)


def test_profile_classification():
    seen = classify_workload_for_model("J2-IT-ResNetInf-GPT2Train", "2x2_train_resnet_gpt2")
    one = classify_workload_for_model("J2-IT-LlamaInf-ResNetTrain", "2x2_train_resnet_gpt2")
    two = classify_workload_for_model("J2-IT-LlamaInf-BloomTrain", "2x2_train_resnet_gpt2")
    assert seen["Benchmark_Type"] == "Seen-Profile Baseline"
    assert one["Benchmark_Type"] == "One Held-Out Profile"
    assert two["Benchmark_Type"] == "Two Held-Out Profiles"


def test_profile_suite():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        names = [
            "J2-IT-ResNetInf-GPT2Train",
            "J2-IT-LlamaInf-ResNetTrain",
            "J2-IT-LlamaInf-BloomTrain",
        ]
        catalog = {}
        for name in names:
            path = root / f"{name}.ini"
            path.write_text("[x]\n", encoding="utf-8")
            catalog[name] = path
        suite = build_profile_suite(catalog, model_id="2x2_train_resnet_gpt2")
        assert set(suite["Benchmark_Type"]) == {
            "Seen-Profile Baseline", "One Held-Out Profile", "Two Held-Out Profiles"
        }


def test_multipart_and_recovery():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = b"abc" * 1000
        parts = []
        for index, block in enumerate([payload[:1000], payload[1000:2000], payload[2000:]], start=1):
            path = root / f"data.zip.part{index:03d}"
            path.write_bytes(block)
            parts.append(path)
        output = reassemble_multipart_file(parts, root / "data.zip", expected_size_bytes=len(payload))
        assert output.read_bytes() == payload
        prior = root / "prior"; current = root / "current"
        (prior / "a").mkdir(parents=True)
        (prior / "a" / "result.json").write_text('{"ok":true}', encoding="utf-8")
        restored = restore_prior_outputs(prior, current)
        assert restored == ["a/result.json"]
        assert json.loads((current / "a" / "result.json").read_text())["ok"] is True


def test_validation_aggregate_and_html():
    frame = pd.DataFrame({
        "Case_ID":["x","x"], "Model":["a","a"], "Candidate_Type":["Optimized","Optimized"],
        "Candidate_Rank":[1,1], "Seed":[30,31], "Actual_Both_Pass":[True,False],
        "Actual_P90_Tracking":[0.2,0.4], "Actual_Max_Pj":[0.08,0.12],
        "Actual_Full_Objective":[1.0,2.0],
    })
    agg = aggregate_validation(frame)
    assert int(agg.iloc[0]["Runs"]) == 2
    assert bool(agg.iloc[0]["All_Seeds_Pass"]) is False
    html = build_report(title="Test", tables={"Aggregate":agg})
    assert "<style>" in html and "Aggregate" in html and "#T_" not in html



def test_explicit_ratio_starts():
    workload = WorkloadSpec(
        path="synthetic.ini",
        job_names=["ResNetInf", "GPT2Train"],
        mix=np.asarray([
            [100.0, 300.0, 10.0, 20.0, 4.0, 1.0],
            [120.0, 320.0, 12.0, 22.0, 4.0, 1.0],
        ]),
    )
    experiment = ExperimentSpec(
        path="experiment.ini", server_count=1000, utilization=0.8,
        idle_watts=100.0, simulation_duration_seconds=3600,
        random_seed=20, iso_file_path="iso.csv", iso_signal_start_hour=0,
    )
    bounds = calculate_pr_bounds(workload, r_lower_kw_per_server=0.01)
    settings = OptimizationSettings(
        starts=12, iterations=1,
        explicit_ratio_starts=tuple((p, r) for p in [0.6,0.8,1.0] for r in [0.2,0.4,0.6,0.8]),
        illegal_start_policy="project", r_over_p_max=0.6,
        weight_min_fraction_of_equal=0.6, weight_max_multiple_of_equal=1.8,
    )
    manifest = prepare_ratio_starts(
        workload=workload, experiment=experiment, bounds=bounds,
        settings=settings, ratio_starts=settings.explicit_ratio_starts,
    )
    assert len(manifest) == 12
    assert (~manifest["Skipped"]).all()
    assert manifest["Projected"].dtype == bool


def test_prep_manifest_schema_and_package_deduplication():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = b"PK\x03\x04" + b"x" * 5000
        parts = []
        for index, start in enumerate(range(0, len(payload), 1000), start=1):
            part = root / f"bundle.zip.part{index:03d}"
            part.write_bytes(payload[start:start + 1000])
            parts.append(part)
        manifest = {
            "archive_name": "bundle.zip",
            "archive_size_bytes": len(payload),
            "archive_sha256": __import__("hashlib").sha256(payload).hexdigest(),
            "parts": [
                {"name": part.name, "size_bytes": part.stat().st_size, "sha256": sha256_file(part)}
                for part in parts
            ],
        }
        manifest_path = root / "bundle_upload_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        rebuilt = reassemble_from_manifest(manifest_path, search_dir=root)
        assert rebuilt.read_bytes() == payload

        source = root / "same.txt"
        source.write_text("one physical file", encoding="utf-8")
        archive = package_paths(root / "dedupe.zip", [source, source, root], root=root)
        import zipfile
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
        assert names.count("same.txt") == 1


def test_scenario_aliases():
    case = ScenarioDefinition.from_mapping({
        "case_id":"x", "workload":"w.ini", "experiment":"e.ini",
        "N":1000 if False else None, "utilization":0.8,
        "pbar":0.5, "r":0.2, "weights":"[0.5,0.5]",
    })
    assert case.workload_config == "w.ini"
    assert case.initial_weights == (0.5,0.5)


if __name__ == "__main__":
    test_profile_classification()
    test_profile_suite()
    test_multipart_and_recovery()
    test_validation_aggregate_and_html()
    test_explicit_ratio_starts()
    test_scenario_aliases()
    test_prep_manifest_schema_and_package_deduplication()
    print("test_flexdc_inference_orchestration: PASS")
