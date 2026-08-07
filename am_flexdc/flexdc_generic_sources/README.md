# CONDOR–FlexDC Generic Behavior-Model Framework

This directory contains the reusable CONDOR–FlexDC behavior-model framework derived from the active **Model V3** implementation.

Its purpose is to preserve the complete V3 training, inference, optimization, FlexDC-validation, recovery, W&B, reporting, and artifact workflows while removing unnecessary model-version naming and path assumptions.

## Important: what "generic" means

The **core behavior-model framework is generic**.

It is not restricted to the J=2 experiment.

The core model and inference stack support:

- variable numbers of jobs \(J\);
- arbitrary FlexDC workload INI files;
- arbitrary supported server counts and utilizations;
- arbitrary valid scheduler-weight vectors matching the workload size;
- configurable Pbar/R operating regions;
- checkpoint-driven model loading rather than V2/V3 filename rewriting;
- different dataset sizes and sweep densities;
- single-model or multi-model inference;
- real FlexDC validation across configurable simulator seeds.

However, this directory also contains **optional experiment-specific tooling for the J=2 model-family holdout study**.

That distinction is important.

---

# Framework Layers

The code is best understood as two layers.

## Layer 1: Generic CONDOR–FlexDC behavior-model framework

These components are intended to support Model V3 and future behavior models.

```text
FlexDC dataset
      |
      v
generic behavior-model training
      |
      v
checkpoint / training artifact
      |
      v
generic prediction + optimization
      |
      v
real FlexDC validation
```

The generic core includes:

```text
data_center_model_flexdc_behavior.py
flexdc_behavior_training_utilities.py
flexdc_behavior_inference_utilities.py
flexdc_behavior_evaluation.py
flexdc_inference_orchestration.py
flexdc_colab_orchestration.py
flexdc_presentation.py

flexdc_behavior_predict_one.py
flexdc_behavior_optimize_one.py
flexdc_behavior_end_to_end_eval.py
```

These modules are not intrinsically tied to J=2.

## Layer 2: J=2 profile-holdout experiment

The following components belong specifically to the controlled J=2 generalization experiment:

```text
flexdc_profile_split.py
flexdc_profile_holdout.py
test_flexdc_profile_holdout.py
model_specs.json
prepare_flexdc_training_dataset.py
```

They define and evaluate the four J=2 model-family holdout experiments:

```text
3x1_llama_holdout
3x1_resnet_holdout
2x2_train_resnet_gpt2
2x2_train_llama_bloom
```

The J=2 experiment is used to study whether the V3-derived architecture can generalize to job types from unseen model families.

It is **not the replacement training methodology for Model V3 or future full-scale behavior models**.

---

# Relationship to Model V3

Model V3 is the active full behavior-model baseline.

The generic framework retains the important V3 design choices:

- mask-aware Set Transformer workload encoder;
- variable-length workload sets;
- 13 engineered features per job;
- 12 global FlexDC features;
- PMA plus masked-mean workload pooling;
- residual global prediction trunk;
- direct prediction of:
  - mean normalized tracking behavior;
  - p90 normalized tracking behavior;
  - one QoS probability \(P_j\) per real job;
- analytical reconstruction of the FlexDC objective;
- training-only normalization;
- context/feasibility/boundary-aware sampling;
- QoS-weighted loss;
- warmup plus cosine learning-rate scheduling;
- gradient clipping;
- early stopping;
- multiple checkpoint roles;
- interruption recovery;
- validation-only checkpoint selection;
- untouched final test evaluation.

The generic code should therefore be viewed as a **maintainable implementation of the V3 behavior-model framework**, not as a separate V4 model.

---

# Variable Job Count

The core behavior model is not hard-coded to two jobs.

The Set Transformer receives a padded workload tensor and a validity mask:

```text
[B, J, F_job]
```

where:

```text
B     = batch size
J     = number of job types in that workload
F_job = per-job feature dimension
```

Padding is masked throughout attention, pooling, outputs, losses, and metrics.

The model therefore supports different workload sizes as long as the input dataset and workload configuration contain the corresponding job metadata.

The inference and optimization utilities likewise derive the job count from the selected FlexDC workload and create the appropriate number of scheduler weights.

---

# Generic Model Inputs

## Per-job features

Each real job is represented by 13 engineered features derived from:

- minimum and maximum job power;
- power range;
- minimum and maximum runtime;
- runtime ratio;
- QoS threshold;
- job size;
- QoS headroom;
- scheduler weight;
- relative scheduler weight;
- approximate node allocation;
- queue pressure.

## Global features

The model also receives 12 global features describing the selected FlexDC operating point and workload context.

These include transformed information derived from:

- Pbar;
- reserve R;
- P/R relationships;
- utilization;
- server count;
- number of jobs.

The feature construction is shared between training and inference.

---

# Model Outputs

The neural network predicts simulator behavior rather than directly predicting one final objective value.

Direct outputs are:

```text
log mean normalized tracking behavior
log p90 normalized tracking behavior
per-job QoS probabilities P_j
```

The remaining FlexDC quantities are reconstructed analytically.

Conceptually:

```text
predicted simulator behavior
        |
        +--> tracking quantities
        |
        +--> per-job QoS probabilities
        |
        v
analytical FlexDC objective reconstruction
```

This keeps the surrogate tied to interpretable simulator behavior and makes the reconstructed objective differentiable for downstream optimization.

---

# Main Notebooks

There are three full-parity notebooks.

## 1. Training and Evaluation

```text
FlexDC_Generic_Full_Parity_Training_and_Evaluation.ipynb
```

This notebook preserves the complete V3 training workflow:

- local or Colab execution;
- repository cloning/reuse;
- dataset acquisition;
- W&B;
- training and resume;
- checkpoint recovery;
- validation-only checkpoint selection;
- locked evaluation;
- plots and reports;
- optional post-freeze optimization;
- artifact packaging.

### Current limitation

The **current notebook wrapper is configured around the J=2 profile-holdout experiment**.

It currently expects:

```python
MODEL_ID = "..."
```

to resolve to one of the four J=2 holdout definitions.

Therefore:

> The underlying training/model implementation is generic, but this notebook's current dataset-selection section is J=2-specific.

For the active full Model V3 dataset or a future full-scale dataset, the same generic model/training modules can be used, but the notebook should load the dataset's native train/validation/test assignment directly instead of applying the J=2 profile-holdout splitter.

The J=2 `MODEL_ID` mechanism should not be interpreted as a requirement of the model architecture.

---

## 2. Inference and FlexDC Validation

```text
FlexDC_Generic_Full_Parity_Inference_Validation.ipynb
```

This notebook is substantially more general.

It supports:

- one or multiple model artifacts;
- arbitrary compatible checkpoints;
- arbitrary workload configurations;
- variable job count;
- Predict One;
- multi-start constrained optimization;
- top-k candidate selection;
- optimization trajectories;
- multiple simulator seeds;
- real FlexDC validation;
- predicted-versus-actual comparison;
- custom benchmark cases;
- resumable batch evaluation;
- identical-point comparisons across models;
- W&B and HTML reports.

### J=2 defaults

The notebook currently ships with a J=2 example:

```python
SCENARIO = "J2-IT-ResNetInf-GPT2Train"
```

and explicit example weight limits:

```python
WEIGHT_MIN = 0.3
WEIGHT_MAX = 0.7
```

Those are **example settings for the J=2 experiment**, not architectural restrictions.

For a different \(J\), provide the appropriate workload and scheduler-weight domain or use the generic relative weight-bound controls.

---

## 3. Paired / Multi-Model Comparison

```text
FlexDC_Generic_Full_Parity_Paired_Comparison.ipynb
```

This notebook preserves the detailed paired-comparison workflow used during development.

It supports:

- two or more model artifacts;
- identical FlexDC runtime reconstruction;
- identical prediction points;
- shared optimizer settings;
- shared simulator seeds;
- exact shared starting points;
- separate validation of each model's optimized candidate;
- resumable batch comparison;
- W&B;
- presentation reports;
- output packaging.

The orchestration is reusable, but several default cases, workload names, gradient-config paths, and comparison rounds are inherited from the experiments from which the notebook was reconstructed.

They should be treated as **editable experiment presets**, not assumptions of the generic model.

---

# J=2 Profile-Holdout Experiment

The bundle includes the J=2 experiment because it was used to evaluate the generalization behavior of the V3-derived architecture.

The experiment uses four model families:

```text
ResNet
GPT-2
Llama
Bloom
```

with two job types per family:

```text
Inference
Training
```

giving eight job types total.

The master experiment contains all 28 valid two-job workloads:

```text
6 inference + inference
6 training + training
16 inference + training
-----------------------
28 workloads
```

Two holdout designs are evaluated.

## 3x1

Three model families are visible during training and one complete family is held out.

Examples:

```text
train:   ResNet, GPT-2, Bloom
holdout: Llama
```

or:

```text
train:   GPT-2, Llama, Bloom
holdout: ResNet
```

## 2x2

Two model families are visible and two complete families are held out.

Examples:

```text
train:   ResNet, GPT-2
holdout: Llama, Bloom
```

or:

```text
train:   Llama, Bloom
holdout: ResNet, GPT-2
```

The J=2 splitter classifies test workloads into:

```text
seen workload
one unseen job
two unseen jobs
```

This machinery exists to answer a particular research question about architectural transfer.

It should not be applied automatically to future full-scale training datasets.

---

# Dataset Preparation

The bundle contains:

```text
prepare_flexdc_training_dataset.py
```

This script has two responsibilities.

## Generic responsibilities

It can:

- discover FlexDC exact-plan worker outputs;
- verify exact `Plan_Row_ID` coverage;
- combine results and diagnostics;
- recover authoritative plan metadata;
- collapse dynamic weight columns;
- reconstruct missing workload metadata;
- validate behavior-model input schema;
- validate objective reconstruction;
- create a master training-ready dataset.

Those operations are generally useful.

## J=2-specific responsibilities

It also:

- imports the J=2 profile splitter;
- generates the four predefined holdout datasets;
- writes J=2 model-specific manifests and audits.

Therefore the script as currently packaged is **not a completely experiment-neutral dataset builder**.

For future Model V3-style training, the generic combine/audit portion can be retained while the J=2 profile-split stage should be omitted or replaced by the desired dataset split policy.

---

# Training Data for Full Models

For Model V3 and future full-scale behavior models, dataset generation should happen in the FlexDC repository.

The current V3 dataset methodology uses dense simulator coverage across:

```text
Pbar
R
scheduler weights
server count
utilization
workload
```

with structured, Sobol, and physical-boundary sampling.

The CONDOR-FLEXDC repository consumes the resulting audited simulator dataset.

The intended separation is:

```text
FlexDC repository
    |
    |  simulation-plan generation
    |  simulation
    |  dataset auditing
    v
training-ready simulator dataset
    |
    v
CONDOR-FLEXDC generic behavior-model framework
    |
    |  training
    |  evaluation
    |  inference
    |  optimization
    v
trained behavior model
```

The J=2 experiment is only one optional evaluation path within this broader framework.

---

# Source Modules

| Module | Scope | Purpose |
|---|---|---|
| `data_center_model_flexdc_behavior.py` | Generic | Variable-J masked Set Transformer |
| `flexdc_behavior_training_utilities.py` | Generic | Features, normalization, sampling, loss, training, metrics, checkpoints |
| `flexdc_behavior_inference_utilities.py` | Generic | Checkpoint loading, prediction, optimization, FlexDC validation |
| `flexdc_behavior_evaluation.py` | Mostly generic | Evaluation tables, boundary metrics, constrained ranking |
| `flexdc_inference_orchestration.py` | Generic core + experiment presets | Scenarios, seeded experiments, resume, suites |
| `flexdc_colab_orchestration.py` | Generic | Clone/setup, upload, extraction, artifact handling |
| `flexdc_presentation.py` | Generic | Colab-safe HTML reporting |
| `flexdc_behavior_predict_one.py` | Generic | Standalone prediction CLI |
| `flexdc_behavior_optimize_one.py` | Generic | Standalone optimizer CLI |
| `flexdc_behavior_end_to_end_eval.py` | Generic | Prediction/optimization plus optional simulator validation |
| `flexdc_profile_split.py` | **J=2 experiment** | Four model-family holdout split definitions |
| `flexdc_profile_holdout.py` | **J=2 experiment** | Holdout-specific evaluation helpers |
| `test_flexdc_profile_holdout.py` | **J=2 experiment** | Tests for the four holdout definitions |

---

# Standalone CLIs

The framework contains three reusable command-line tools.

## Predict One

```text
flexdc_behavior_predict_one.py
```

Loads a compatible checkpoint and predicts behavior for one FlexDC operating point.

## Optimize One

```text
flexdc_behavior_optimize_one.py
```

Runs constrained multi-start gradient optimization through the learned surrogate.

## End-to-End Evaluation

```text
flexdc_behavior_end_to_end_eval.py
```

Combines surrogate optimization with optional real FlexDC validation.

These commands use the generic checkpoint/model interface and are not dependent on the J=2 profile-holdout model IDs.

---

# Checkpoints and Artifacts

Training artifacts are intended to be self-contained.

They may contain:

- model checkpoints;
- model configuration;
- normalization statistics;
- behavior-model constants;
- training/evaluation metrics;
- predictions;
- plots;
- HTML reports;
- source snapshots;
- repository manifests;
- FlexDC runtime configuration files required for later validation.

Inference discovers the model configuration from the artifact/checkpoint rather than requiring V2-to-V3 filename copying or source-text replacement.

This is one of the main reasons for maintaining the generic framework.

---

# Preserved Full-Parity Functionality

The package was reconstructed from the working V3 training, V3 inference, and paired-comparison workflows.

The goal was to remove version-specific coupling **without removing useful capabilities**.

Preserved functionality includes:

### Training

- clone/reuse;
- local and Colab execution;
- direct and multipart dataset upload;
- W&B modes;
- complete V3 architecture;
- grouped splits;
- balanced sampling;
- training-only normalization;
- learning-rate schedule;
- gradient clipping;
- early stopping;
- checkpoint roles;
- resume;
- recovery;
- validation-only checkpoint selection;
- untouched final test;
- reporting and artifact packaging.

### Inference

- checkpoint/artifact discovery;
- Predict One;
- exact and random starts;
- constrained optimization;
- top-k candidates;
- trajectories;
- safety calibration;
- real FlexDC validation;
- multi-seed validation;
- predicted-versus-actual tables;
- batch/resume;
- multi-model comparisons;
- W&B;
- packaging.

### Paired comparison

- common runtime;
- common starts;
- common simulator seeds;
- independent optimized candidates;
- real simulator comparison;
- resumable rounds;
- reports and export.

See:

```text
FEATURE_PARITY_MATRIX.md
```

for the detailed mapping.

---

# Validation Status

See:

```text
PACKAGE_VALIDATION_REPORT.md
```

for the exact tests that were executed.

A major source of confusion is that much of the validation uses the **J=2 profile-holdout dataset**.

That does **not** mean the underlying Set Transformer requires J=2.

The J=2 synthetic repository was used because it provides a deterministic end-to-end fixture for testing:

- variable workload metadata;
- exact-plan reconstruction;
- profile splitting;
- train/validation/test leakage;
- packaging;
- inference;
- optimization;
- multi-model comparison.

Separate structural tests validate variable job count, padding, masks, and permutation behavior of the generic model.

The validation report demonstrates software/workflow integrity. It does not claim scientific performance for future models or datasets.

---

# What This Package Does Not Claim

This framework does not claim that:

- the J=2 sampling design should replace Sweep V3;
- the four J=2 holdout models are the future production models;
- J must equal two;
- a profile-holdout split is required for normal training;
- synthetic smoke tests establish scientific model accuracy;
- surrogate feasibility removes the need for simulator validation.

FlexDC remains the ground-truth system used for scientific validation.

---

# Recommended Use

## For active Model V3 work

Use the generic model/training/inference modules while preserving the V3 dataset and behavior-model methodology.

Do not apply the J=2 profile holdout unless that is the experiment being performed.

## For future full behavior models

Start from:

```text
data_center_model_flexdc_behavior.py
flexdc_behavior_training_utilities.py
flexdc_behavior_inference_utilities.py
flexdc_behavior_evaluation.py
flexdc_inference_orchestration.py
```

and use an audited FlexDC dataset with an appropriate train/validation/test policy.

The current full-parity training notebook should be generalized beyond its J=2 `MODEL_ID` selection before being treated as the canonical future full-model training notebook.

## For the J=2 architecture experiment

Use:

```text
prepare_flexdc_training_dataset.py
flexdc_profile_split.py
flexdc_profile_holdout.py
FlexDC_Generic_Full_Parity_Training_and_Evaluation.ipynb
```

with one of the four defined model-family holdout specifications.

---

# Repository Relationship

The two repositories have different responsibilities.

## FlexDC

```text
https://github.com/amenon871/FlexDC
```

owns:

- simulation;
- workload/config generation;
- exact sweep plans;
- simulator dataset generation;
- simulator-side auditing.

## CONDOR-FLEXDC

```text
https://github.com/NetherMoon/CONDOR-FLEXDC
```

owns:

- the learned behavior model;
- training;
- model evaluation;
- inference;
- differentiable optimization;
- surrogate-versus-simulator validation.

---

# Supporting Documentation

Read these files for more detail:

| File | Purpose |
|---|---|
| `COLAB_QUICKSTART.md` | How to run the notebooks in Colab |
| `FEATURE_PARITY_MATRIX.md` | Capability mapping from the original working notebooks |
| `SOURCE_AUDIT.md` | Source lineage and reconstruction decisions |
| `PACKAGE_VALIDATION_REPORT.md` | Tests and smoke executions actually performed |

---

# Summary

The simplest way to think about this directory is:

```text
                 GENERIC CORE
                     |
       +-------------+-------------+
       |                           |
       v                           v
full V3/future model work     J=2 holdout experiment
       |                           |
 arbitrary J                   fixed J=2 study
 native dataset split          3x1 / 2x2 splits
 future production             architecture evaluation
```

**The core model is generic.  
The J=2 profile-holdout experiment is not.  
Model V3 remains the active full-model baseline.**