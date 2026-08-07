# CONDOR: Learning Data Center Models for Demand Response

This repository contains the original **CONDOR** implementation from:

> **Learning a Data Center Model for Efficient Demand Response**  
> Quentin Clark, Fatih Acun, Andreas Paschyllidis, and Ayse K. Coskun  
> HotCarbon 2024, July 9, 2024, Santa Cruz, CA.

It also contains subsequent **CONDOR-FlexDC behavior-modeling research** under:

```text
am_flexdc/
```

---

## Overview

**CONDOR** (Cost-Optimization Neural network for Data center Operational demand Response) uses a learned surrogate model to optimize data-center participation in demand response.

The original workflow trains a neural network on simulated data-center behavior and then differentiates through the trained model to optimize controllable parameters such as:

- expected average power;
- reserve capacity;
- scheduler allocation across workloads.

The repository now contains two related research tracks:

```text
Original CONDOR
    |
    | surrogate modeling + differentiable DR optimization
    |
    v
HotCarbon 2024 implementation

CONDOR-FlexDC
    |
    | FlexDC-generated simulator data
    | behavior prediction
    | differentiable optimization
    |
    v
am_flexdc/
```

---

## Original CONDOR

The original CONDOR code, datasets, and experiments remain in the repository for reproducing and extending the HotCarbon 2024 work.

For the original workflow, use the directories outside:

```text
am_flexdc/
```

---

## CONDOR-FlexDC Extension

The directory:

```text
am_flexdc/
```

contains later work adapting the CONDOR approach to the **FlexDC** data-center demand-response simulator.

The current full-model baseline is **FlexDC Behavior Model V3**.

Unlike earlier experiments that directly predicted cost terms, the current model predicts underlying FlexDC behavior, including:

- tracking behavior;
- per-job QoS behavior.

The FlexDC objective is then reconstructed analytically, allowing gradients to be used to optimize:

```text
Pbar
R
scheduler weights
```

Optimized candidates are ultimately validated using the real FlexDC simulator.

For the full current workflow, architecture, training methodology, inference, and validation documentation, see:

```text
am_flexdc/README.md
```

---

## Generic Behavior-Model Framework

A reusable version of the current training and inference infrastructure is maintained under:

```text
am_flexdc/generic_full_parity/
```

It contains reusable support for:

- variable-job-count behavior models;
- training and checkpoint recovery;
- inference;
- differentiable optimization;
- FlexDC validation;
- model comparison;
- Colab workflows.

This framework is derived from the active V3 implementation and is intended to support future models and experiments.

It is **not limited to J=2**.

---

## J=2 Generalization Experiment

The repository also contains an isolated **J=2 model-family holdout experiment**.

This experiment evaluates whether the V3-derived model architecture can generalize to job types from model families that were not present during training.

It is an architecture/generalization study and should not be confused with the main V3 full-model training methodology.

---

## Related FlexDC Repository

FlexDC simulation and dataset generation are maintained separately at:

```text
https://github.com/amenon871/FlexDC
```

The division of responsibility is:

```text
FlexDC
    simulation
    workload/config generation
    dataset generation
    simulator-side auditing

CONDOR-FLEXDC
    model training
    evaluation
    inference
    differentiable optimization
    simulator validation
```

FlexDC remains the ground-truth simulator for scientific validation.

---

## Repository Structure

```text
CONDOR-FLEXDC/
│
├── am_flexdc/
│   ├── train/                  # active Model V3 implementation
│   ├── generic_full_parity/    # reusable behavior-model framework
│   ├── data/
│   ├── models/
│   └── README.md
│
├── ...                         # original CONDOR code and experiments
│
└── README.md
```

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/NetherMoon/CONDOR-FLEXDC.git
cd CONDOR-FLEXDC
```

For the original CONDOR implementation, use the original repository directories and notebooks.

For current FlexDC behavior-model work, start with:

```text
am_flexdc/README.md
```

For the reusable training/inference framework, start with:

```text
am_flexdc/generic_full_parity/README.md
```

---

## Publications

### Original CONDOR

> Quentin Clark, Fatih Acun, Andreas Paschyllidis, and Ayse K. Coskun.  
> **Learning a Data Center Model for Efficient Demand Response.**  
> HotCarbon 2024, July 9, 2024, Santa Cruz, CA.

### FlexDC

FlexDC simulator and publication information are available in:

```text
https://github.com/amenon871/FlexDC
```

---

## Authors

Original CONDOR authors:

- Quentin Clark
- Fatih Acun
- Andreas Paschyllidis
- Ayse K. Coskun

Original main contact:

```text
Quentin Clark
q.clark@mail.utoronto.ca
```

Subsequent FlexDC behavior-modeling extensions are maintained under `am_flexdc/`.