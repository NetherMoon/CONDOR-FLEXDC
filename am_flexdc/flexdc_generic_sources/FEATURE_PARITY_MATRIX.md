# Feature Parity Matrix

This matrix records how the repaired generic notebooks map to the three working notebooks used as the baseline. Genericization changes names, paths, and model selection; it does not intentionally remove workflow capabilities.

## Training notebook parity

Baseline: `am_unified_model_training_wandb_colab_paths_configured_objective_flexdc_behavior_v3 (2)(1).ipynb`

| Original capability | Repaired location | Status |
|---|---|---|
| Colab/local environment controls | Training §0 | Preserved and expanded |
| Clone/reuse CONDOR-FLEXDC | Training §3 | Preserved |
| Clone/reuse FlexDC | Training §3 | Added for post-freeze validation and artifact self-containment |
| Dirty-checkout protection and optional update/reclone | Training §0–3 | Preserved/expanded |
| Optional Google Drive path | Training §4 | Preserved, disabled by default |
| Direct upload and local path modes | Training §2 and §5 | Preserved/expanded |
| Large-dataset multipart upload | Training §5 | Added; SHA-256 and size verified |
| Dataset discovery and required-file checks | Training §5–7 | Preserved/expanded |
| Automatic install of locally generated workloads/configs | Training §6 | Added through the dataset runtime bundle |
| Compile original and generic sources | Training §8 | Preserved |
| Mask/padding/permutation/split structural tests | Training §8 | Preserved |
| W&B login, force relogin, offline/disabled modes | Training §10 | Preserved/expanded |
| Full V3 architecture and 13/12 engineered features | Training §9, §11, source modules | Preserved |
| Preassigned grouped split and training-only normalization | Training §12 | Preserved |
| Natural/context/feasibility/boundary sampler | Training §11–12 | Preserved |
| QoS-weighted loss and boundary multipliers | Training §11, §15 | Preserved |
| Warmup + cosine schedule | Training §11, §15 | Preserved |
| Gradient clipping and early stopping | Training §11, §15 | Preserved |
| Latest/best-loss/best-objective/best-feasibility/final checkpoints | Training §15 | Preserved |
| Resume from checkpoint | Training §0, §15 | Preserved |
| Interrupted-runtime recovery cell | Training §16 | Preserved |
| Validation-only checkpoint comparison | Training §17 | Preserved |
| Locked checkpoint train + validation evaluation | Training §18 | Preserved |
| One-time untouched test evaluation | Training §19 | Preserved |
| W&B metric/table logging | Training §14, §21 | Preserved/expanded |
| Training curves | Training §22 | Preserved |
| Artifact ZIP and optional Colab download | Training §24 | Preserved/expanded |
| Generic choice among four profile-holdout models | Training §0, §7 | Added |
| Seen/one-held/two-held profile evaluation | Training §20 | Added, not substituted for native metrics |
| Boundary/failure/constrained-ranking/seed-stability tables | Training §20 | Added |
| Correct constrained regret semantics | Evaluation module | Fixed: infeasible selections do not receive regret |
| Optional post-freeze optimizer + multi-seed FlexDC validation | Training §23 | Added using the same inference orchestration module |
| Presentation-ready HTML without raw Pandas Styler CSS | Training §9, §20, §24 | Fixed/expanded |

## Single-/multi-model inference notebook parity

Baseline: `am_unified_inference_flexdc_behavior_v3_presentation_tables (2).ipynb`

| Original capability | Repaired location | Status |
|---|---|---|
| Colab/local controls and logging | Inference §0 | Preserved/expanded |
| Clone/reuse both repositories | Inference §3 | Preserved |
| Optional local checkout overrides | Inference §0, §3 | Preserved |
| Install generic sources beside repository sources | Inference §3 | Added; original sources are not overwritten |
| Artifact upload/path/repository discovery | Inference §4 | Preserved/expanded |
| One model without renaming V2 files | Inference §4–6 | Fixed |
| Multiple models for comparisons | Inference §4, §19 | Preserved/expanded |
| Runtime reconstruction from training artifact | Inference §4 | Preserved/expanded |
| Restore prior result ZIP/directory | Inference §4, §10, batch sections | Preserved |
| Checkpoint-role selection | Inference §0, §4 | Preserved |
| Scenario preset and custom workload | Inference §8 | Preserved/expanded |
| All P/R, weight, safety, optimizer, and seed controls | Inference §0, §8–9, §12 | Preserved |
| Predict One | Inference §10 | Preserved |
| Standalone Predict One command | Inference §11 | Preserved |
| Optimize-only multi-start workflow | Inference §12 | Preserved |
| Exact/ratio/random/near-equal/high-P-low-R starts | Inference §8, §12 | Preserved/expanded |
| Trajectory plots | Inference §13 | Preserved |
| Top-k distinct candidates | Inference §12 | Preserved |
| Starting-point and top-k real FlexDC validation | Inference §14 | Preserved |
| Multiple simulator seeds and all-seeds aggregation | Inference §14–15 | Expanded |
| Predicted-versus-actual tables | Inference §15, §20 | Preserved/expanded |
| Safety-margin calibration | Inference §9 | Preserved |
| Fixed seen/one-held/two-held suite | Inference §16 | Added and executable |
| Custom legacy rounds | Inference §17 | Preserved as configurable cases |
| Resumable all-workload benchmark | Inference §18 | Preserved/expanded |
| V2/V3-style identical-point comparison | Inference §19 | Generalized to any loaded models |
| Shared deterministic-start comparison | Inference §19 | Preserved/expanded |
| External/Fatih/reference-point CSV | Inference §19 | Preserved/expanded |
| Presentation-ready output | Inference §20 | Preserved; raw CSS leak fixed |
| W&B tables and summaries | Inference §7, §21 | Preserved/expanded |
| Optional full orchestrator CLI command | Inference §22 | Preserved |
| Complete output ZIP | Inference §23 | Preserved/expanded |

## Paired comparison notebook parity

Baseline: `CONDOR_FlexDC_Paired_Comparison_V3_V4_Optimization_Fixed (1).ipynb`

| Original capability | Repaired location | Status |
|---|---|---|
| Two-artifact setup | Paired §0, §4 | Preserved and generalized to 2+ models |
| Clone/update both repositories | Paired §3 | Preserved |
| Editable FlexDC install without dependency replacement | Paired §3 | Preserved using `--no-deps` |
| Shared runtime reconstruction | Paired §4 | Preserved |
| Restore completed outputs | Paired §8A | Preserved |
| Exact explicit HTML | Paired §6 | Preserved/fixed |
| Reusable Round 2/4/V4 workloads | Paired §7 | Preserved |
| Exact 12 Fatih P/R start pairs | Paired §0, §8 | Preserved |
| Same optimizer settings and simulator seeds across models | Paired §8 | Preserved |
| Predict One with one shared actual simulator point | Paired §9 | Preserved |
| Optimize One | Paired §10 | Preserved |
| Rounds 1–4 | Paired §11–14 | Preserved |
| Round 5 separately labeled/disabled by default | Paired §15 | Preserved rationale; executable when enabled |
| Candidate validation separately per model | Paired helper in §8 | Preserved and corrected |
| Resumable all-workload paired optimization | Paired §16 | Preserved |
| W&B and complete packaging | Paired §17 | Preserved/expanded |

## Dependency-file parity

| File | Role |
|---|---|
| `flexdc_behavior_training_utilities.py` | Generic V3 behavior labels, features, normalization, sampler, training, metrics, checkpoints, recovery |
| `data_center_model_flexdc_behavior.py` | Mask-aware Set Transformer and per-job QoS head |
| `flexdc_behavior_inference_utilities.py` | Generic checkpoint load, prediction, exact/shared starts, optimization, FlexDC validation |
| `flexdc_inference_orchestration.py` | Scenario catalog, seeded experiments, resume, batch/suite execution, multi-seed validation |
| `flexdc_colab_orchestration.py` | Clone/update, verified upload/extraction, multipart reconstruction, runtime install, checkpoint discovery, packaging |
| `flexdc_behavior_evaluation.py` | Native/profile/boundary/failure/constrained-ranking/seed metrics and W&B/HTML output |
| `flexdc_presentation.py` | Explicit HTML reports that render correctly in Colab |
| `prepare_flexdc_training_dataset.py` | Exact-plan combine/audit/split plus self-contained runtime and multipart bundle creation |
