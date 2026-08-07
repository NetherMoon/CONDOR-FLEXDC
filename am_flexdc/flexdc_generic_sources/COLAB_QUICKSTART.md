# Colab Quick Start

## A. Prepare the data on the PC

Place this package's `prepare_flexdc_training_dataset.py` and `flexdc_generic_sources/` in the FlexDC root. Run:

```powershell
python -u prepare_flexdc_training_dataset.py `
  --plan-file flexdc_sweep_plan_j2_pairwise.csv `
  --optimization-dir src/peacsim/output/optimization `
  --prefix j2_pairwise_master `
  --out-dir flexdc_j2_pairwise_training_dataset `
  --output-prefix j2_pairwise `
  --models all `
  --gradient-config configs/gradient_descent/gradient_descent_j2_pairwise_rsr.ini `
  --cluster-config configs/cluster/cluster.ini `
  --colab-part-size-mib 20
```

The output includes:

```text
j2_pairwise_profile_holdout_training_bundle.zip
j2_pairwise_profile_holdout_training_bundle.zip.part001
j2_pairwise_profile_holdout_training_bundle.zip.part002
...
j2_pairwise_profile_holdout_training_bundle_upload_manifest.json
```

Use the multipart files plus manifest when the full ZIP is too large for a reliable browser upload. Each part is verified and the reconstructed ZIP is checked against the original SHA-256.

## B. Train one model

Open:

```text
FlexDC_Generic_Full_Parity_Training_and_Evaluation.ipynb
```

Upload this code bundle ZIP when prompted. For the dataset, upload all `.partNNN` files and the matching `upload_manifest.json` together. A complete ZIP or direct CSV is also supported.

Choose one model in Section 0:

```python
MODEL_ID = "3x1_llama_holdout"
MODEL_ID = "3x1_resnet_holdout"
MODEL_ID = "2x2_train_resnet_gpt2"
MODEL_ID = "2x2_train_llama_bloom"
```

The notebook automatically:

1. verifies/installs dependencies without replacing PyTorch;
2. clones or reuses CONDOR-FLEXDC and FlexDC;
3. reconstructs and verifies the dataset;
4. installs the generated workloads, experiment, gradient, cluster, plan, and referenced runtime files into the FlexDC checkout;
5. compiles sources and runs structural tests;
6. trains/resumes/recovers;
7. compares checkpoints on validation only;
8. freezes the selected checkpoint;
9. evaluates the untouched test set and profile-holdout categories;
10. exports a self-contained artifact ZIP.

The final training artifact already contains the generated FlexDC runtime files. No separate J2 config upload is required in the inference notebook.

## C. Run one-model inference and real simulator validation

Open:

```text
FlexDC_Generic_Full_Parity_Inference_Validation.ipynb
```

Upload this code bundle and one training artifact ZIP. Use:

```python
ARTIFACT_MODE = "upload_zip"
RUN_FLEXDC_VALIDATION = True
SIMULATOR_SEEDS = [30, 31, 32]
```

The notebook clones both repositories, reconstructs the model/runtime, predicts, optimizes, validates the starting point and top-k candidates in FlexDC, and produces predicted-versus-actual tables.

For multiple models, use:

```python
ARTIFACT_MODE = "multi_upload"
```

## D. Run the full paired comparison battery

Open:

```text
FlexDC_Generic_Full_Parity_Paired_Comparison.ipynb
```

Upload two or more training artifacts. Enable only the desired blocks:

```python
RUN_PREDICT_ONE = True
RUN_OPTIMIZE_ONE = True
RUN_ROUND_1 = True
RUN_ROUND_2 = True
RUN_ROUND_3 = True
RUN_ROUND_4 = True
RUN_ROUND_5 = False
RUN_ALL_WORKLOAD_PAIRED_OPTIMIZATION = False
```

All loaded models use the same explicit P/R starts, optimizer settings, candidate rules, and simulator seeds. A fixed physical point is simulated once per seed and shared across model predictions; model-specific optimized candidates are validated separately.
