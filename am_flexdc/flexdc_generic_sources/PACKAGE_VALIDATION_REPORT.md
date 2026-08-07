# Package Validation Report

Validation date: 2026-08-06

## Static validation

- All Python source files compile.
- All three full-parity notebooks parse as valid notebook JSON.
- Every notebook code cell parses as valid Python.
- Static full-parity guards verify clone/setup, recovery, checkpoint, W&B, optimization, real-validation, batch, comparison, and packaging controls are present in their respective notebooks.
- Dataset-preparation and all three standalone CLI entry points expose `--help` successfully.

## Structural tests

Passed all inherited behavior-model tests:

- feature dimensions;
- variable job count and masks;
- padding/batch invariance;
- permutation equivariance;
- zero gradient into padding;
- sampler and learning-rate schedule;
- preassigned split and repeat-group normalization;
- repeat-seed leakage rejection.

Passed all profile-holdout tests for the four model definitions:

| Model | Seen workloads | One-held workloads | Two-held workloads |
|---|---:|---:|---:|
| `3x1_llama_holdout` | 15 | 12 | 1 |
| `3x1_resnet_holdout` | 15 | 12 | 1 |
| `2x2_train_resnet_gpt2` | 6 | 16 | 6 |
| `2x2_train_llama_bloom` | 6 | 16 | 6 |

Passed orchestration tests covering:

- scenario aliases and profile categories;
- fixed profile suite construction;
- deterministic multipart reconstruction;
- output recovery;
- seed aggregation;
- Colab-safe HTML rendering;
- exact normalized ratio starts.

## Dataset builder end-to-end smoke

A synthetic FlexDC-compatible repository with 28 J=2 workloads and 56 contexts passed:

- exact-plan/result/diagnostic coverage;
- merge and objective-lineage checks;
- all four model splits and audits;
- self-contained runtime-bundle creation;
- generated workload/experiment/gradient/cluster/plan inclusion;
- upload-part generation;
- manifest-based reassembly;
- reconstructed size and SHA-256 equality;
- reconstructed ZIP integrity.

## Training notebook execution smoke

The full-parity training notebook was executed cell-by-cell with a synthetic dataset, one epoch, and a reduced model. It completed:

- package and dataset acquisition;
- runtime installation;
- model-specific dataset resolution;
- training-only normalization and sampler audit;
- model construction and real-batch forward pass;
- training and checkpoint creation;
- validation-only checkpoint comparison;
- frozen train/validation evaluation;
- untouched test evaluation;
- profile, boundary, failure, constrained-ranking, and seed tables;
- presentation report;
- self-contained artifact ZIP.

The structural-test cell was tested separately rather than nested inside this in-process smoke because nested PyTorch subprocesses can hang in the container. The exact test scripts passed independently.

## Inference notebook execution smoke

The full-parity inference notebook was executed cell-by-cell with a real smoke checkpoint and synthetic FlexDC-compatible configs. It completed:

- existing-directory package mode;
- local repository overrides;
- training-artifact extraction;
- automatic runtime-config installation;
- checkpoint discovery/load;
- safety calibration;
- Predict One;
- constrained multi-start optimization;
- identical-point comparison;
- presentation report;
- final output ZIP.

Real FlexDC execution was disabled in this smoke because the synthetic repository does not contain the actual simulator. The real-validation path is covered by the original workflow logic and unit-level orchestration tests, but a scientific simulator run requires the user's real FlexDC checkout/data.

## Paired notebook execution smoke

The full-parity paired notebook was executed cell-by-cell with two artifact instances and a synthetic repository. It completed:

- multi-artifact load;
- shared runtime reconstruction;
- source integration;
- exact HTML layer;
- generation of Round 2, Round 4, V4, and duration experiment configs;
- restore/resume initialization;
- disabled-round control flow;
- paired report and final ZIP.

The expensive paired simulator rounds were not run against the synthetic repository.

## Standalone CLI smoke

Completed:

- `flexdc_behavior_predict_one.py`;
- `flexdc_behavior_optimize_one.py`;
- `flexdc_behavior_end_to_end_eval.py` without real simulator execution.

## Not claimed

This package validation does not claim scientific performance for the real four-model campaign. It does not substitute for running optimized candidates through the actual FlexDC simulator over the selected seed set. The notebooks contain those workflows; their results must come from the user's real checkout and data.
