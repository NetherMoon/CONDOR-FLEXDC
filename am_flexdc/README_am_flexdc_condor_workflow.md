# FlexDC → CONDOR Training, Inference, and Validation Workflow

This folder contains the FlexDC/CONDOR surrogate-modeling work. The current active experiment is the **CONDOR-label model trained on FlexDC-generated data**.

The older **FlexDC-objective model** is preserved as legacy. Do not mix the two workflows.

---

## 1. Current active experiment

### What this model is

The current model uses:

- **FlexDC** as the simulator/data generator.
- **CONDOR** as the label/objective convention and neural-network architecture.

For each FlexDC pilot row, the loader reconstructs the original CONDOR-style targets:

| CONDOR target | Constructed from FlexDC row |
|---|---|
| `cost_power` | `0.0003 * (P_actual_watts - R_actual_watts)` |
| `cost_error` | `Mtrack_Error_MeanAbs_Watts / 1000` |
| `cost_qos` | `0.8 * sum(SoftPlus(60 * (QoS_Delay_Probability - 0.1)))` |

Then the released CONDOR implementation scaling is applied:

```text
scaled_cost_power = 120 * cost_power / server_count
scaled_cost_error = 200 * cost_error / server_count
scaled_cost_qos   = cost_qos / workload_mix_size
```

The model inputs remain CONDOR-style:

```text
[P, R, server_count, utilization, workload_mix_size]
+ workload matrix with [pmin, pmax, Tmin, Tmax, QoS threshold, job size, weight]
```

The input values come from the FlexDC pilot dataset.

This is **not** the same as the earlier FlexDC-objective model that predicted:

```text
M_RSR, C_track, C_Qos
```

---

## 2. Recommended active folder layout

Use this organization inside `comder-main/am_flexdc`:

```text
am_flexdc/
├── data/
│   ├── initial_data/
│   │   ├── traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv
│   │   └── traditional_iso16_fullpilot_AQA_combined_grid_search_diagnostics.csv
│   └── legacy_flexdc_objective/
│       ├── flexdc_all_data.csv
│       └── convert_flexdc_to_condor.ipynb
│
├── models/
│   ├── am_condor_flexdc_labels_v1_state_dict.pt
│   └── legacy_flexdc_objective/
│       └── am_flexdc_condor_flexdc_objective_v1_state_dict.pt
│
├── results/
│   ├── condor_inference_w1_n1000_u060/
│   └── legacy_flexdc_objective/
│       ├── current_model_w1_n1000_u060_equal/
│       └── weighted_objective_w1_n1000_u060_t2_q2_r50/
│
└── train/
    ├── data_center_model.py
    ├── am_condor_flexdc_training_utilities.py
    ├── am_condor_flexdc_model_training_wandb_colab.ipynb
    ├── am_condor_flexdc_inference.py
    ├── am_compare_condor_flexdc_validation.py
    └── legacy_flexdc_objective/
        ├── am_flexdc_training_utilities.py
        ├── am_flexdc_model_training_notebook.ipynb
        ├── am_flexdc_model_training_notebook_wandb.ipynb
        ├── am_flexdc_inference.py
        ├── am_flexdc_inference_weighted_objective.py
        └── am_plot_flexdc_model_surfaces.py
```

The active code path is:

```text
train/data_center_model.py
train/am_condor_flexdc_training_utilities.py
train/am_condor_flexdc_model_training_wandb_colab.ipynb
train/am_condor_flexdc_inference.py
train/am_compare_condor_flexdc_validation.py
```

---

## 3. Training in Google Colab

Training is expected to run in Colab because CPU training is slow locally.

### 3.1 Colab runtime setup

In Colab:

1. Open `am_condor_flexdc_model_training_wandb_colab.ipynb`.
2. Select `Runtime → Change runtime type`.
3. Choose a GPU runtime.

The notebook installs W&B with:

```python
!pip -q install wandb
```

Then it logs in with:

```python
import wandb
wandb.login()
```

### 3.2 Files to upload to `/content`

For training only, upload:

```text
data_center_model.py
am_condor_flexdc_training_utilities.py
traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv
```

For inference from Colab as well, also upload:

```text
am_condor_flexdc_inference_colab.py
traditional_iso16_fullpilot_AQA_combined_grid_search_diagnostics.csv
```

### 3.3 Training cells

Run the notebook cells in order through the training/evaluation section.

The main training call is:

```python
sim_model, train_loss_record, heldout_loss_record = train_model(
    sim_model,
    epochs=150,
    lr=1e-4,
    batch_size=512,
    verbose=True,
    cross_validate=True,
    data_file_path=DATA_FILE,
    wandb_run=run,
)
```

This uses:

```text
70% train / 30% held-out split
seed = 0
AdamW optimizer
MSE loss
batch size = 512
learning rate = 1e-4
epochs = 150
```

### 3.4 Output of training

The notebook saves:

```text
/content/am_condor_flexdc_labels_v1_state_dict.pt
```

Download this model and place it in:

```text
am_flexdc/models/am_condor_flexdc_labels_v1_state_dict.pt
```

Do not save or rely on the full Python model object. Use the `state_dict` file.

### 3.5 W&B logging

The notebook logs:

```text
train_loss
heldout_loss
heldout MAE/RMSE/R2 metrics
```

Local Colab `wandb/` folders are disposable logs/caches. The real run is stored online in W&B.

Do not commit the local `wandb/` folder.

---

## 4. Running CONDOR-style inference

Inference uses the frozen trained model to optimize the inputs `P`, `R`, and workload weights `w`.

The model does **not** predict `P`, `R`, or `w`. The model predicts costs. Gradients of the predicted objective are used to update `P`, `R`, and `w`.

Current inference objective:

```text
0.05 * predicted_cost_power
+ 0.7 * predicted_cost_error
+ 2.0 * predicted_cost_qos
```

These weights come from the released CONDOR example and can be changed later as an experiment.

Run from:

```bash
cd "comder-main/am_flexdc/train"
```

Example:

```bash
python am_condor_flexdc_inference.py \
  --results-csv "../data/initial_data/traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv" \
  --diagnostics-csv "../data/initial_data/traditional_iso16_fullpilot_AQA_combined_grid_search_diagnostics.csv" \
  --model-file "../models/am_condor_flexdc_labels_v1_state_dict.pt" \
  --out-dir "../results/condor_inference_w1_n1000_u060" \
  --workload "W1-train" \
  --server-count 1000 \
  --utilization 0.60 \
  --weight-sample-id 0 \
  --iterations 150 \
  --lr 0.01 \
  --power-weight 0.05 \
  --error-weight 0.7 \
  --qos-weight 2.0
```

This writes:

```text
results/condor_inference_w1_n1000_u060/optimized_candidate.json
results/condor_inference_w1_n1000_u060/optimization_trajectory.csv
results/condor_inference_w1_n1000_u060/optimization_comparison_before_validation.csv
```

Use `optimized_candidate.json` to get the optimized `P`, `R`, and `w` for real FlexDC validation.

---

## 5. Running FlexDC validation

After inference, run the starting and optimized configurations in FlexDC.

Run these from the FlexDC simulator repository:

```bash
cd "flexdc-sim-main/src/peacsim"
```

The data wizard should be the reproducibility-patched version, where `random.seed(...)` and `np.random.seed(...)` are called before `init_job_table(...)`.

### 5.1 Starting configuration

Fill values from `optimized_candidate.json`:

```bash
python -u am_data_extraction_wizard.py \
  --gradient-config ../../configs/gradient_descent/gradient_descent.ini \
  --experiment-config ../../configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini \
  --cluster-config ../../configs/cluster/cluster.ini \
  --policy-name AQA \
  --job-config ../../configs/workload/W1-train.ini \
  --output-dir condor_labels_w1_n1000_u060_start \
  --utilization-values 0.60 \
  --auto-workload-pr-sweep false \
  --pbar-kw-per-server-values "<P_START>" \
  --r-kw-per-server-values "<R_START>" \
  --weight-vectors "<W_START_COMMA_SEPARATED>" \
  --node-count-control true \
  --pbar-lower-factor 0.9 \
  --pbar-upper-factor 1.0 \
  --pr-upper-factor 1.2 \
  --pr-chunk-index 0 \
  --pr-num-chunks 1
```

### 5.2 Optimized configuration

```bash
python -u am_data_extraction_wizard.py \
  --gradient-config ../../configs/gradient_descent/gradient_descent.ini \
  --experiment-config ../../configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini \
  --cluster-config ../../configs/cluster/cluster.ini \
  --policy-name AQA \
  --job-config ../../configs/workload/W1-train.ini \
  --output-dir condor_labels_w1_n1000_u060_optimized \
  --utilization-values 0.60 \
  --auto-workload-pr-sweep false \
  --pbar-kw-per-server-values "<P_OPT>" \
  --r-kw-per-server-values "<R_OPT>" \
  --weight-vectors "<W_OPT_COMMA_SEPARATED>" \
  --node-count-control true \
  --pbar-lower-factor 0.9 \
  --pbar-upper-factor 1.0 \
  --pr-upper-factor 1.2 \
  --pr-chunk-index 0 \
  --pr-num-chunks 1
```

Each validation command should say:

```text
number of P/R pairs = 1
number of weight vectors = 1
Total simulator runs to execute: 1
```

If it says more than one run, stop immediately and check the command.

---

## 6. Comparing model prediction to FlexDC validation

Copy the two one-row FlexDC result files into a convenient location, or refer to them directly from their output folders.

Run from `am_flexdc/train`:

```bash
python am_compare_condor_flexdc_validation.py \
  --candidate-json "../results/condor_inference_w1_n1000_u060/optimized_candidate.json" \
  --inference-comparison-csv "../results/condor_inference_w1_n1000_u060/optimization_comparison_before_validation.csv" \
  --start-results-csv "<PATH_TO_START_GRID_SEARCH_RESULTS.csv>" \
  --optimized-results-csv "<PATH_TO_OPTIMIZED_GRID_SEARCH_RESULTS.csv>" \
  --out-csv "../results/condor_inference_w1_n1000_u060/condor_end_to_end_validation.csv"
```

The final table should include:

```text
Pbar
R
weights
predicted weighted objective
actual weighted objective
tracking error
QoS violation ratio
```

Report this as:

```text
CONDOR-label training and input-gradient optimization using FlexDC-generated configurations and measured simulation outcomes.
```

Do not describe it as the FlexDC full-objective model.

---

## 7. Legacy FlexDC-objective workflow

The earlier legacy model predicted:

```text
M_RSR
C_track
C_Qos
```

and optimized:

```text
M_RSR + C_track + C_Qos
```

That was useful for prototyping, but it is not the current CONDOR-aligned direction from Kerim.

Keep its files under `legacy_flexdc_objective/` for reference only.

---

## 8. File inventory

### Active data files

| File | Purpose |
|---|---|
| `data/initial_data/traditional_iso16_fullpilot_AQA_combined_grid_search_results.csv` | Main FlexDC pilot results. Used for training, inference context, and CONDOR-label construction. |
| `data/initial_data/traditional_iso16_fullpilot_AQA_combined_grid_search_diagnostics.csv` | FlexDC diagnostics. Used by inference/comparison where needed. |

### Active model files

| File | Purpose |
|---|---|
| `models/am_condor_flexdc_labels_v1_state_dict.pt` | Current trained CONDOR-label model state dictionary. |

### Active training/inference files

| File | Purpose |
|---|---|
| `train/data_center_model.py` | Released CONDOR neural architecture. Shared by all current training/inference. |
| `train/am_condor_flexdc_training_utilities.py` | Current dataset loader/training/evaluation utilities. Reads FlexDC results but constructs CONDOR-style labels. |
| `train/am_condor_flexdc_model_training_wandb_colab.ipynb` | Main Colab training notebook with W&B logging. Use this for future training runs. |
| `train/am_condor_flexdc_inference.py` | Current CONDOR-label inference script. Loads frozen model and optimizes `P`, `R`, and `w`. |
| `train/am_compare_condor_flexdc_validation.py` | Compares model predictions with actual one-run FlexDC validation outputs. |
| `train/aqa_parsing_utilities.py` | Legacy/support utility from CONDOR. Keep only if imported by other scripts. |

### Active result folders

| Folder | Purpose |
|---|---|
| `results/condor_inference_w1_n1000_u060/` | Current inference output for W1, 1000 servers, utilization 0.60. Contains optimized candidate and trajectory. |

### Legacy FlexDC-objective files

| File or folder | Purpose |
|---|---|
| `data/legacy_flexdc_objective/flexdc_all_data.csv` | Converted dataset for old FlexDC-objective model. Not used by current CONDOR-label workflow. |
| `data/legacy_flexdc_objective/convert_flexdc_to_condor.ipynb` | Old converter notebook. Archived. |
| `models/legacy_flexdc_objective/am_flexdc_condor_flexdc_objective_v1_state_dict.pt` | Old model that predicted `M_RSR`, `C_track`, and `C_Qos`. |
| `train/legacy_flexdc_objective/am_flexdc_training_utilities.py` | Old FlexDC-objective training utilities. |
| `train/legacy_flexdc_objective/am_flexdc_model_training_notebook.ipynb` | Old FlexDC-objective training notebook. |
| `train/legacy_flexdc_objective/am_flexdc_model_training_notebook_wandb.ipynb` | Old FlexDC-objective W&B notebook. |
| `train/legacy_flexdc_objective/am_flexdc_inference.py` | Old FlexDC-objective inference script. |
| `train/legacy_flexdc_objective/am_flexdc_inference_weighted_objective.py` | Old weighted-objective experiment script. |
| `train/legacy_flexdc_objective/am_plot_flexdc_model_surfaces.py` | Old surface plot script for FlexDC-objective model. |
| `results/legacy_flexdc_objective/current_model_w1_n1000_u060_equal/` | Old FlexDC-objective inference result. |
| `results/legacy_flexdc_objective/weighted_objective_w1_n1000_u060_t2_q2_r50/` | Old weighted-objective inference result. |

### Disposable files/folders

These should not be committed or used as source of truth:

```text
__pycache__/
.ipynb_checkpoints/
wandb/
.config/
sample_data/
```

---

## 9. Safety notes

- Do not overwrite `am_condor_flexdc_labels_v1_state_dict.pt` unless intentionally retraining a new version.
- Do not mix the legacy FlexDC-objective model with the current CONDOR-label inference scripts.
- Always validate any optimized candidate by running FlexDC once for the starting configuration and once for the optimized configuration.
- For reporting, distinguish:
  - model-predicted CONDOR weighted objective;
  - actual FlexDC-run reconstructed CONDOR weighted objective;
  - FlexDC full objective, if shown separately.
