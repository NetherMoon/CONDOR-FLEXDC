# CONDOR-FlexDC Behavior Modeling

This directory contains the CONDOR/FlexDC surrogate-modeling research developed on top of the original CONDOR codebase.

The project uses **FlexDC as the ground-truth data-center demand-response simulator** and a **CONDOR-derived Set Transformer** as a differentiable surrogate for FlexDC behavior.

The current full-model baseline is **FlexDC Behavior Model V3**.

This directory also contains:

- the reusable generic behavior-model framework derived from V3;
- inference and differentiable optimization infrastructure;
- FlexDC validation and model-comparison workflows;
- an isolated J=2 model-family holdout experiment used to study architecture generalization;
- tests and supporting research utilities.

Older V1, V2, original-CON­DOR-target, raw-objective, and intermediate unified-model work is no longer part of the active workflow and is kept locally under the gitignored `archive/` directory when historical access is useful.

---

# 1. Project Status

The current project should be understood as four related pieces.

| Component | Status | Purpose |
|---|---|---|
| **Behavior Model V3** | **Active full-model baseline** | Main CONDOR-derived FlexDC surrogate architecture and training methodology |
| **Generic Full-Parity Framework** | **Active reusable framework** | Version-neutral training, inference, optimization, comparison, recovery, reporting, and packaging infrastructure derived from V3 |
| **J=2 Model-Family Holdout Experiment** | **Active isolated experiment** | Tests whether the architecture generalizes to job types from unseen model families |
| V1/V2/old unified workflows | Legacy | Preserved locally only when useful for research history |

**Model V3 is not a historical model. It is the active baseline from which future full-scale models are being developed.**

The J=2 study is not a replacement for V3 training. It is a controlled experiment for evaluating the model architecture and its ability to transfer to unseen workload/job profiles.

---

# 2. Repository Relationship

The project is split across two repositories.

## FlexDC

Repository:

```text
https://github.com/amenon871/FlexDC
```

FlexDC owns the **simulation and dataset-generation side**:

- data-center simulation;
- workload definitions;
- experiment configurations;
- Pbar/R operating-point generation;
- scheduling-weight sweeps;
- exact simulation plans;
- parallel simulator execution;
- simulation-output auditing;
- training-dataset generation.

## CONDOR-FLEXDC

Repository:

```text
https://github.com/NetherMoon/CONDOR-FLEXDC
```

`am_flexdc/` owns the **learned-model side**:

- behavior-model architecture;
- feature construction;
- normalization;
- model training;
- checkpointing;
- model evaluation;
- inference;
- differentiable optimization;
- surrogate-versus-FlexDC validation;
- generalization experiments;
- model comparison;
- reporting.

The intended pipeline is:

```text
             FLEXDC REPOSITORY
                    |
                    | generate exact simulation plan
                    | run FlexDC
                    | audit simulator outputs
                    v
          training-ready dataset
                    |
                    v
           CONDOR-FLEXDC / am_flexdc
                    |
                    | train surrogate
                    | evaluate model
                    | optimize Pbar, R, and w
                    v
            candidate operating point
                    |
                    v
             real FlexDC validation
```

FlexDC is always the ground-truth simulator.

---

# 3. Active Model: FlexDC Behavior Model V3

The active surrogate does **not** directly predict the final FlexDC objective.

Instead, it learns underlying simulator behavior and reconstructs the known objective analytically.

## Direct neural outputs

For each configuration, the model predicts:

```text
log(mean normalized tracking error + floor)

log(p90 normalized tracking error + floor)

one QoS delay/violation probability P_j
for every real job type j
```

The model therefore learns:

```text
FlexDC operating point
+
data-center context
+
variable-length workload
+
scheduler allocation
            |
            v
predicted tracking behavior
+
predicted per-job QoS behavior
```

Known FlexDC cost equations are then applied outside the neural network.

---

# 4. Model Architecture

Model V3 retains the original CONDOR idea of treating the workload as a **set rather than a fixed ordered vector**.

The workload encoder is a mask-aware Set Transformer.

Conceptually:

```text
job 1 features ─┐
job 2 features ─┤
job 3 features ─┤
      ...       ├─> Set Transformer
job J features ─┘        |
                         +--> PMA pooling
                         |
                         +--> masked-mean pooling
                                  |
                                  v
                         workload representation

global FlexDC features ----------+
                                  |
                                  v
                         residual MLP trunk
                            /           \
                           /             \
                  tracking heads      per-job QoS head
```

## Variable job count

The architecture is **not hard-coded to J=2 or J=4**.

Workloads are padded within a batch and accompanied by a validity mask.

Padding is excluded from:

- attention;
- pooling;
- outputs;
- losses;
- metrics.

The model therefore supports variable numbers of real job types as long as the corresponding workload information is available.

The J=2 experiment uses the same architecture with two real job tokens. J=2 is an experiment configuration, not an architectural restriction.

---

# 5. Input Features

Model V3 uses engineered physical and scheduling features instead of only the original raw CONDOR inputs.

## Per-job features

Each real job token contains 13 features derived from:

```text
minimum job power
maximum job power
power range
minimum runtime
maximum runtime
runtime ratio
QoS threshold
job size
QoS headroom
scheduler weight
weight relative to equal allocation
approximate allocated nodes
queue pressure
```

These features describe both the underlying job and how the scheduler is currently allocating resources to it.

## Global features

The model also receives 12 global features derived from:

```text
Pbar
R
Pbar - R
Pbar + R
R / Pbar
Pbar ratio
R ratio
utilization
server count
log server count
inverse job count
scaled job count
```

Feature normalization statistics are calculated using **training data only** and saved with the model checkpoint/artifact for later inference.

---

# 6. FlexDC Objective Reconstruction

The neural model predicts behavior.

Known cost terms are reconstructed analytically.

For tracking:

```text
mean_tracking_prediction
        |
        v
Mtrack
```

The simulator RSR monetary term is then reconstructed as:

```text
M_RSR = Simulator_Power_Cost + Mtrack
```

Tracking penalty:

```text
C_track =
psi * SoftPlus(
    mu * (p90_tracking - tracking_threshold)
)
```

Per-job QoS penalty:

```text
C_QoS =
beta * sum_j SoftPlus(
    rho * (P_j - QoS_threshold)
)
```

The final FlexDC objective is:

```text
Full Objective =
M_RSR
+ C_track
+ C_QoS
```

The current V3 behavior-model constants use:

```text
tracking threshold = 0.30
QoS threshold      = 0.10

psi  = 1
mu   = 10

beta = 20
rho  = 2
```

These constants must remain consistent between:

- FlexDC dataset generation;
- model training;
- inference;
- optimization;
- simulator validation.

A dataset should be generated correctly in the first place rather than repaired afterward with a different objective convention.

---

# 7. Feasibility

The primary feasibility rule is:

```text
p90 normalized tracking error <= 0.30
```

and

```text
max_j(P_j) <= 0.10
```

Both conditions must hold.

Therefore:

```text
feasible
=
tracking pass
AND
every real job satisfies the QoS probability limit
```

Using only mean QoS can hide a single badly violating job, which is why the model predicts one \(P_j\) for each real job type.

The evaluation workflow also pays particular attention to configurations near the decision boundaries.

Typical diagnostic regions are:

```text
Tracking boundary:
0.20 <= p90 <= 0.40

QoS boundary:
0.05 <= max(P_j) <= 0.15
```

---

# 8. Active V3 Training Methodology

Model V3 introduced several changes over the earlier experimental model versions.

## Dataset handling

The active pipeline supports:

- preassigned train/validation/test splits;
- grouping of repeated simulator seeds;
- protection against repeat-seed leakage across splits;
- base-configuration normalization so repeated seeds do not receive extra importance merely because they were repeated.

## Training sampling

Training can balance examples across:

- workload/context;
- feasibility;
- tracking boundary;
- QoS boundary.

Validation and test metrics remain unweighted representations of the held-out data.

## Optimization

The training loop uses:

- AdamW;
- learning-rate warmup;
- cosine learning-rate decay;
- gradient clipping;
- early stopping;
- resumable optimizer state.

## Checkpoints

Several checkpoint roles are retained:

```text
latest
best loss
best objective
best feasibility
```

Checkpoint selection is performed using validation data.

The final test set should remain frozen until checkpoint/model selection is complete.

## Experiment tracking

The training workflow supports Weights & Biases for:

- run metadata;
- loss curves;
- validation metrics;
- checkpoint artifacts;
- experiment comparison.

Local `wandb/` directories are not a source of truth and should not be committed.

---

# 9. Evaluation

Global model accuracy alone is not sufficient for this project.

The model is intended to support **constrained optimization**, so evaluation must include both regression quality and safety/feasibility behavior.

Important metrics include:

## Tracking

```text
R²
MAE
p90 tracking error
```

## Per-job QoS

```text
per-job P_j R² / error
max-P_j error
```

## Reconstructed objective

```text
R²
MAE
Spearman rank correlation
```

Rank quality matters because the model is used to choose between candidate operating points.

## Feasibility

```text
precision
recall
F1
accuracy
false-feasible rate
```

False-feasible predictions are especially important because the optimizer can exploit regions that the surrogate incorrectly believes are safe.

## Context-level evaluation

Metrics should also be reported by:

```text
workload
server count
utilization
generalization category
boundary subset
```

Strong global metrics must not hide complete failure in a particular workload or operating regime.

---

# 10. Active V3 Files

The original active V3 implementation is maintained under:

```text
am_flexdc/train/
```

The core V3 files are:

```text
data_center_model_flexdc_behavior_v3.py
am_flexdc_behavior_training_utilities_v3.py
test_flexdc_behavior_training_v3.py
am_unified_model_training_wandb_colab_paths_configured_objective_flexdc_behavior_v3.ipynb
```

### `data_center_model_flexdc_behavior_v3.py`

Defines the active masked Set Transformer behavior model.

### `am_flexdc_behavior_training_utilities_v3.py`

Contains the active:

- dataset handling;
- engineered features;
- normalization;
- sampling;
- losses;
- objective reconstruction;
- training loop;
- evaluation;
- checkpoint logic.

### `test_flexdc_behavior_training_v3.py`

Tests important model invariants including:

- feature dimensions;
- variable job count;
- mask handling;
- padding invariance;
- permutation behavior;
- zero influence/gradient from padding;
- training/split behavior.

### V3 training notebook

```text
am_unified_model_training_wandb_colab_paths_configured_objective_flexdc_behavior_v3.ipynb
```

This is the original full V3 Colab training workflow.

It remains useful as the direct V3 reference implementation even as reusable functionality is migrated into the generic framework.

---

# 11. Generic Full-Parity Framework

The directory:

```text
am_flexdc/generic_full_parity/
```

contains a reusable framework reconstructed from the working V3 training, inference, optimization, validation, and paired-comparison workflows.

The purpose of this framework is to remove:

- V2/V3 filename coupling;
- hard-coded Colab paths;
- source-text rewriting;
- one-off checkpoint assumptions;

without removing useful functionality from the original workflows.

## Important: "generic" does not mean "J=2"

The **core generic model framework supports variable J**.

It is intended to support:

- active Model V3;
- future full behavior models;
- arbitrary compatible FlexDC workloads;
- future experiments.

However, the package also contains some **J=2-specific experiment modules and notebook defaults** because the J=2 holdout study was used as an end-to-end validation target while the generic package was being built.

See:

```text
generic_full_parity/README.md
```

for the exact distinction between the generic core and J=2 experiment tooling.

## Main generic notebooks

```text
FlexDC_Generic_Full_Parity_Training_and_Evaluation.ipynb

FlexDC_Generic_Full_Parity_Inference_Validation.ipynb

FlexDC_Generic_Full_Parity_Paired_Comparison.ipynb
```

The framework preserves functionality including:

- repository cloning/reuse;
- local and Colab operation;
- upload/recovery;
- W&B;
- training;
- resume;
- checkpoint comparison;
- inference;
- multi-start constrained optimization;
- real FlexDC validation;
- multi-seed validation;
- batch evaluation;
- paired model comparison;
- HTML reports;
- artifact packaging.

---

# 12. J=2 Model-Family Holdout Experiment

The J=2 work is a **separate architecture/generalization experiment**.

It should not be confused with the active V3 full-model training dataset.

The experiment asks:

> How well does the V3-derived architecture generalize to job types belonging to model families that were completely absent during training?

## Model families

The experiment uses:

```text
ResNet
GPT-2
Llama
Bloom
```

Each family has two job types:

```text
Inference
Training
```

giving eight total job types.

Each workload contains exactly two jobs.

The complete master set contains:

```text
6 inference + inference workloads
6 training + training workloads
16 inference + training workloads
---------------------------------
28 total workloads
```

## Holdout designs

### 3x1

Three model families are visible during training.

One complete model family is held out.

Both the inference and training job types of the held-out family are unseen.

### 2x2

Two model families are visible during training.

Two complete model families are held out.

## Test categories

The experiment evaluates:

```text
Seen workload
    both jobs belong to training-visible families

One unseen job
    one job belongs to a held-out family

Two unseen jobs
    both jobs belong to held-out families
```

The experiment is meant to evaluate **architecture transfer**, not define the sampling strategy for future production/full-model training.

J=2-specific profile splitting lives in the generic package rather than in the core V3 architecture.

---

# 13. Inference

Inference uses a frozen behavior model.

The model does not predict the optimization variables themselves.

The optimization variables are:

```text
Pbar
R
scheduler weights w
```

The model predicts the behavior resulting from those variables.

Because the full reconstructed objective is differentiable, gradients can be propagated from:

```text
reconstructed FlexDC objective
        |
        v
predicted behavior
        |
        v
Pbar, R, w
```

to search for improved operating points.

The generic inference stack supports:

- Predict One;
- random and structured starting points;
- multi-start optimization;
- exact feasibility constraints;
- safety margins;
- configurable P/R bounds;
- configurable scheduler-weight bounds;
- top-k candidate selection;
- optimization trajectories.

---

# 14. FlexDC Validation

A surrogate-generated optimum is **not a final scientific result** until it is run through the real FlexDC simulator.

The validation workflow is:

```text
starting point
       |
       +------> surrogate prediction
       |
       +------> real FlexDC run

optimized point
       |
       +------> surrogate prediction
       |
       +------> real FlexDC run
```

The comparison should distinguish clearly between:

```text
predicted tracking
actual FlexDC tracking

predicted per-job QoS
actual FlexDC per-job QoS

predicted objective
actual reconstructed objective

predicted feasibility
actual simulator feasibility
```

Multi-seed validation should be used when simulator stochasticity matters.

The surrogate is an optimization accelerator and behavior approximation. It does not replace FlexDC as the validation authority.

---

# 15. Data Generation

Large FlexDC datasets should be generated in the **FlexDC repository**, not manually constructed inside this repository.

The active full-model baseline is the Sweep V3 methodology.

Sweep V3 uses dense coverage of:

```text
Pbar
R
scheduler weights
workloads
server counts
utilizations
```

with:

- structured P/R sampling;
- Sobol interior sampling;
- physical-boundary sampling;
- broad scheduler-weight coverage;
- repeated seeds;
- audited split assignments.

Future full-scale datasets should use V3 as the baseline methodology unless a new experiment demonstrates that a different sampling design provides sufficient coverage.

The J=2 experiment intentionally used a different, smaller controlled dataset because it was designed to test architecture generalization.

See the FlexDC repository README for the current dataset-generation workflow.

---

# 16. Current `am_flexdc` Organization

The cleaned directory is intended to look approximately like:

```text
am_flexdc/
│
├── README.md
│
├── train/
│   ├── data_center_model_flexdc_behavior_v3.py
│   ├── am_flexdc_behavior_training_utilities_v3.py
│   ├── test_flexdc_behavior_training_v3.py
│   ├── am_unified_model_training_wandb_colab_paths_configured_objective_flexdc_behavior_v3.ipynb
│   └── compatibility files still required by active workflows
│
├── generic_full_parity/
│   ├── FlexDC_Generic_Full_Parity_Training_and_Evaluation.ipynb
│   ├── FlexDC_Generic_Full_Parity_Inference_Validation.ipynb
│   ├── FlexDC_Generic_Full_Parity_Paired_Comparison.ipynb
│   ├── flexdc_generic_sources/
│   └── documentation
│
├── data/
│   └── optional local/current dataset artifacts
│
├── models/
│   └── optional current model artifacts
│
└── archive/
    └── local legacy/research-history files
```

`archive/` is intentionally excluded from Git.

---

# 17. Model and Dataset Artifact Policy

The Git repository should primarily contain **reproducible source code**, not every generated research artifact.

## Commit

Commit:

```text
active model source
training utilities
inference utilities
tests
canonical notebooks
generic framework source
documentation
small reproducibility metadata when useful
```

## Usually do not commit

Do not normally commit:

```text
large training datasets
generated CSV outputs
model checkpoint collections
temporary artifact ZIPs
W&B cache folders
Colab staging files
one-off plots
old models
obsolete notebooks
Python caches
notebook checkpoints
```

Large or temporary artifacts should be stored separately.

---

# 18. Local Archive

The directory:

```text
am_flexdc/archive/
```

is a local research archive and is ignored by Git.

It contains files that may still be useful for understanding development history but should not appear in the active repository.

Examples include:

```text
Behavior Model V1
Behavior Model V2
old unified models
old inference notebooks
old target formulations
old model checkpoints
old datasets
one-off debugging utilities
```

The archive is **not** part of the supported workflow.

Do not import active code from it.

If an active workflow still depends on an archived implementation, that dependency should be migrated into the active source tree.

---

# 19. Temporary V2 Inference Compatibility

Some V2-named inference files may temporarily remain under `train/` because earlier V3 inference workflows generated V3 inference code from those source templates.

These files should not be interpreted as indicating that V2 is still the active model.

The generic full-parity inference implementation removes this design by loading compatible checkpoints directly.

Once the generic inference path has been fully validated against the active V3 artifacts, the remaining V2 inference-template files can also be archived.

New development should not introduce additional dependencies on the V2 implementation.

---

# 20. Testing

At minimum, changes to the active V3 model should run:

```bash
pytest -q am_flexdc/train/test_flexdc_behavior_training_v3.py
```

The generic framework also contains tests for:

- training behavior;
- inference orchestration;
- profile holdout;
- notebook structure/full-parity functionality.

The important model invariants include:

```text
variable-J support
padding invariance
batch invariance
permutation invariance/equivariance
zero padding influence
zero padding gradient
correct repeat grouping
split leakage protection
correct feature dimensions
```

Passing software tests does not replace scientific evaluation against real FlexDC simulations.

---

# 21. Recommended Workflow for Full Model Training

For normal V3/future full-model work:

```text
1. Define the FlexDC experiment.

2. Generate the simulator dataset in the FlexDC repository.

3. Audit the complete dataset before training.

4. Transfer/use the training-ready dataset in CONDOR-FLEXDC.

5. Train the V3-derived behavior surrogate.

6. Select checkpoints using validation data only.

7. Evaluate once on the frozen test set.

8. Run behavior, feasibility, boundary, and ranking diagnostics.

9. Run surrogate optimization.

10. Validate selected candidates with real FlexDC.

11. Compare prediction and simulator results.

12. Preserve the checkpoint, training metadata, source version, and runtime configs required for reproducibility.
```

---

# 22. Recommended Workflow for the J=2 Experiment

For the isolated J=2 model-family holdout experiment:

```text
1. Generate the 28 pairwise workloads in FlexDC.

2. Generate and run the controlled J=2 exact sweep.

3. Build the strict family-holdout dataset.

4. Select one of the defined 3x1 or 2x2 experiments.

5. Train using the same V3-derived architecture.

6. Evaluate:
      seen workloads
      one unseen job
      two unseen jobs

7. Compare 3x1 and 2x2 behavior.

8. Use the experiment to study architectural generalization, not to redefine the main V3 dataset methodology.
```

---

# 23. Reproducibility Rules

When extending this project:

1. **Treat FlexDC as ground truth.**

2. **Treat Model V3 as the active full-model baseline.**

3. Do not mix old CONDOR-label, old direct-objective, V1, or V2 checkpoints with the active V3 pipeline unless performing an explicit comparison.

4. Keep train/validation/test separation strict.

5. Keep repeated simulation seeds grouped.

6. Compute normalization using training data only.

7. Report per-workload/context metrics, not just global metrics.

8. Evaluate near feasibility boundaries.

9. Treat false-feasible predictions as an important safety failure.

10. Validate optimizer-selected points with the real simulator.

11. Keep objective constants consistent across simulation, training, inference, and validation.

12. Preserve the exact source/configuration needed to reproduce important checkpoints.

13. Do not commit temporary generated artifacts merely because they were used in one experiment.

---

# 24. Where to Start

For someone new to this work:

## If you want to understand the active model

Start with:

```text
train/data_center_model_flexdc_behavior_v3.py
train/am_flexdc_behavior_training_utilities_v3.py
```

Then read:

```text
train/test_flexdc_behavior_training_v3.py
```

to understand the invariants the implementation is expected to preserve.

## If you want to train the original active V3 workflow

Start with:

```text
train/am_unified_model_training_wandb_colab_paths_configured_objective_flexdc_behavior_v3.ipynb
```

## If you want the reusable future-facing framework

Start with:

```text
generic_full_parity/README.md
```

and then:

```text
generic_full_parity/FlexDC_Generic_Full_Parity_Training_and_Evaluation.ipynb
generic_full_parity/FlexDC_Generic_Full_Parity_Inference_Validation.ipynb
```

## If you want to generate new FlexDC training data

Go to:

```text
https://github.com/amenon871/FlexDC
```

and follow the dataset-generation documentation there.

---

# 25. Summary

The project has evolved substantially from the original CONDOR-label experiments.

The current research direction is:

```text
                FlexDC simulator
                       |
                       v
          dense audited simulator data
                       |
                       v
        V3-derived masked Set Transformer
                       |
          +------------+-------------+
          |                          |
          v                          v
 tracking behavior             per-job P_j
          |                          |
          +------------+-------------+
                       |
                       v
          analytic objective reconstruction
                       |
                       v
           differentiable optimization
                       |
                       v
              real FlexDC validation
```

The key distinctions are:

- **V3 is the active full-model baseline.**
- **The generic full-parity framework is the reusable implementation derived from V3.**
- **The generic core supports variable job count.**
- **J=2 is a separate architecture/generalization experiment.**
- **V1/V2/old unified work is legacy.**
- **FlexDC remains the scientific ground truth.**