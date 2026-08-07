# Source Audit

## Baseline notebooks reviewed in full

1. `am_unified_model_training_wandb_colab_paths_configured_objective_flexdc_behavior_v3 (2)(1).ipynb`
2. `am_unified_inference_flexdc_behavior_v3_presentation_tables (2).ipynb`
3. `CONDOR_FlexDC_Paired_Comparison_V3_V4_Optimization_Fixed (1).ipynb`
4. The previously delivered generic training bundle and its dependency files.
5. The executed profile-holdout notebook and its saved outputs.

## Repository trees reviewed

- `NetherMoon/CONDOR-FLEXDC`, especially `am_flexdc/train` and its versioned model, training, inference, CLI, end-to-end, and structural-test files.
- `amenon871/FlexDC`, including its experiment/workload/cluster/optimization separation and the data-extraction wizard path used for simulator validation.

## Reconstruction rule

The working notebooks were treated as the operational baseline. A feature was removed only when it was a version-name workaround, and only after its behavior was replaced directly:

- V2-to-V3 filename copying/string replacement was replaced by checkpoint-driven generic modules.
- Hard-coded model-version paths were replaced by artifact/checkpoint discovery.
- Hard-coded dataset sizes were replaced by exact-plan inference and audits.

Everything else was retained or expanded. `FEATURE_PARITY_MATRIX.md` records the section-by-section result.

## Training logic retained

- original V3 Set Transformer architecture and masking semantics;
- direct behavior outputs and analytical objective reconstruction;
- exact feature construction;
- training-only normalization;
- preassigned grouped splits and repeat protection;
- balanced sampler;
- loss weights and boundary emphasis;
- optimizer, warmup/cosine schedule, clipping, early stopping;
- all checkpoint roles, resume, recovery;
- validation-only checkpoint comparison;
- locked train/validation and untouched test evaluation;
- W&B, plots, artifact packaging.

Profile-holdout evaluation was added after the native evaluation. It does not replace the original metrics.

## Inference logic retained

- clone/setup of both repositories;
- artifact acquisition and runtime reconstruction;
- complete scenario and physical-bound controls;
- Predict One and optimize-only flows;
- top-k and trajectories;
- safety calibration;
- real FlexDC validation and predicted-versus-actual output;
- batch/round/all-workload orchestration and recovery;
- identical-point comparisons;
- W&B and presentation/export.

The generic inference notebook accepts one model cleanly. Multi-model comparison is additive.

## Paired-comparison logic retained

- multiple artifacts and shared runtime;
- exact 12 normalized P/R starts;
- identical settings and seeds;
- shared actual result for a fixed point;
- separate validation of model-specific optimized candidates;
- generated Round 2/4/V4 workloads;
- Rounds 1–4;
- separately labeled duration sensitivity;
- resumable all-workload comparison;
- W&B, HTML, and packaging.

## Correctness fixes made during reconstruction

- The preprocessing-only split module no longer imports PyTorch.
- Multipart manifest keys now match the preparation script exactly.
- The dataset ZIP is rejected when it is exactly 25 MiB and invalid, avoiding silent browser truncation.
- Artifact packaging deduplicates the same physical source file.
- Infeasible selected candidates receive no regret value.
- Top-k overlap uses constrained feasible rankings.
- Table rendering uses explicit HTML and no longer emits raw `#T_...` CSS.
- Inference `existing_dir` package mode is functional.
- Training artifacts include generated FlexDC runtime files so inference setup is self-contained.
- Editable FlexDC installation uses `--no-deps` to avoid replacing the active numerical/PyTorch stack.
