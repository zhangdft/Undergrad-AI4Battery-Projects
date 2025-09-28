# Bayesian-Informed Optimization (BIO) Framework
*by Zhongxian Sun (孙中贤)*

A modular framework for closed-loop optimization of battery charging protocols using Bayesian learning, early prediction, and automated experiment generation. Designed to accelerate materials discovery and protocol design under limited experimental budgets.

This pipeline integrates:
- Policy generation
- Data collection & normalization
- Bayesian optimization with gap-based selection (BayesGap)
- XML test file automation
- Simulation & visualization tools

Used in: High-throughput screening of fast-charging battery protocols via early-life performance prediction.

---
## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Workflow Overview](#workflow-overview-in-dual-salt)
  - [1. Generate Policies](#1-generate-policies)
  - [2. Cold Start (Round 0)](#2-cold-start-round-0)
  - [3. Closed-Loop Optimization](#3-closed-loop-optimization)
  - [4. Data Collection](#4-data-collection)
- [Transfer to LHCE](#transfer-to-LHCE)
  - [1. Warm Start (Round 1)](#1-warm-start-round-1)
  - [2: Migrate historical data](#2-migrate-historical-data)
  - [3. Closed-Loop Optimization](#3-closed-loop-optimization-1)
- [Final Notes](#final-notes)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## Getting Started

### Prerequisites
- Python ≥ 3.8
- NEWARE battery cycler platform (for real experiments)
- Optional: MATLAB or NumPy-based solvers if extending physical models

### Installation
Clone the repository
```bash
git clone https://github.com/Dianzhuanshaokao/BIO-Framework.git
cd BIO-Framework
conda create 
```
Install dependencies
```bash
pip install -r requirements.txt
```
## Workflow Overview in Dual-salt
The BIO framework follows a four-stage iterative loop:
### 1. Generate Policies
Confirm that all generated policies fall within hardware limits, here:
| Parameters | min | max | gap | Unit |
|--------|--------|--------|------|------|
| I1  | 0.5  | 2.5  | 0.5 | C |
| I2  | 0.5  | 2.5  | 0.5 | C |
| ton1  | 1  | 10  | 1 | s |
| ton2  | 1  | 10  | 1 | s |
| toff1  | 1  | 10  | 1 | s |
| toff2  | 1  | 10  | 1 | s |

Generate all candidate charging strategies before starting optimization.
```bash
python Generation.py
```
Outputs:
- policies_raw.csv: All 924 unshuffled policies.
- policies_all.csv: Randomly shuffled version used as search space.

These files define the discrete arm space for Bayesian optimization.

### 2. Cold Start (Round 0)
Initialize the first batch of experiments (typically random sampling).
```bash
python BIO_Dual_salt.py --round=0
```
Creates:
- `Dual_salt/data/batch/0.csv`: First 50 randomly selected policies.
- `Dual_salt/data/bounds/0.pkl`: Initial model state (empty history).
- `Dual_salt/data/bounds/0_bounds.pkl`: Posterior bounds (prior only).
- `Tools/File4test/budegt@runs_0/*.xml`: test files

Run these on the NEWARE platform and collect results in `Dual_salt/raw/round0/.` 
There are two forms of testdata: `.csv` or `.nadx`.(see [Data Collection](#4-data-collection))
Data can be processed in correspnding scripts by running 
```bash
python Datacollection_csv.py --round=0 --data_dir='Dual_salt/data' --test_files='Dual_salt/raw'
or
python Datacollection_nadx.py --round=0 --data_dir='Dual_salt/data' --test_files='Dual_salt/raw'
```

### 3. Closed-Loop Optimization
Iteratively optimize policy selection based on observed performance.

For each round `i ≥ 1`:
#### Step 1: Collect and Normalize Dat
After completing experiments in `Tools/File4test/budget@runs_i/*.xml`, preprocess the data:
```bash
python Datacollection_csv.py --round=i --data_dir='Dual_salt/data' --test_files='Dual_salt/raw'
or
python Datacollection_ndax.py --round=i --data_dir='Dual_salt/data' --test_files='Dual_salt/raw'
```
Output:
- `Dual_salt/data/pred/<i>.csv`: normalized early predictions (standardized lifetime).
#### Step 2: Run Optimization
Select next batch using BayesGap algorithm:
```bash
python BIO_Dual_salt.py --round=i
```
Output:
- `Dual_salt/data/batch/<i+1>.csv`: Next batch of 50 policies.
- `Dual_salt/data/bounds/<i>.pkl`: Updated model state.
- `Dual_salt/data/bounds/<i>_bounds.pkl`: Confidence intervals for all arms.
- `Tools/File4test/budget@runs_i+1/*.xml`: Ready-to-upload test files.
Repeat until budget is reached.

### 4. Data Collection
The `Datacollection_csv.py` or `Datacollection_ndax.py` script:
- Reads raw `.csv` or `.ndax` files from `Dual_salt/raw/round<i>/`.
- Handles missing values (NaN → mean imputation).
- Saves cleaned, standardized data to `Dual_salt/data/pred/<i>.csv`

## Transfer to LHCE
To adapt the BIO Framework for use with the LHCE battery testing platform, follow these steps to ensure compatibility in file format, communication protocol, and experimental workflow.
### 1. Warm Start (Round 1)
Select first batch based on 4 rounds optimization (see [Dual_salt optimization](#workflow-overview-in-dual-salt)):

- After completing experiments in LHCE with test files in ``Tools/File4test/budegt@runs_4/*.xml``, store the data in `LHCE/raw/round1` and preprocess the data:
```bash
python Datacollection_csv.py --round=1 --data_dir='LHCE/data' --test_files='LHCE/raw'
or
python Datacollection_ndax.py --round=1 --data_dir='LHCE/data' --test_files='LHCE/raw'
```
Output:
- `LHCE/data/pred/1.csv`: normalized early predictions (standardized lifetime).
### 2: Migrate historical data
The historical `.csv` file should be migrated:
```bash
cp 'Dual_salt/data/batch/4.csv' 'LHCE/data/batch'
mv 'LHCE/data/batch/4.csv' 'LHCE/data/batch/1.csv'
```
### 3: Closed-Loop Optimization
Select next batch using BayesGap algorithm:
```bash
python BIO_LHCE.py --round=1
```
Creates:
- `LHCE/data/batch/2.csv`: First 50 randomly selected policies.
- `LHCE/data/bounds/1.pkl`: Initial model state (empty history).
- `LHCE/data/bounds/1_bounds.pkl`: Posterior bounds (prior only).
- `Tools/File4test/budegt@runs_5/*.xml`: test files

Run these on the NEWARE platform and collect results in `LHCE/raw/round2/.` 
Data can be processed in correspnding scripts by running 
```bash
python Datacollection_csv.py --round=2 --data_dir='LHCE/data' --test_files='LHCE/raw'
or
python Datacollection_nadx.py --round=2 --data_dir='LHCE/data' --test_files='LHCE/raw'
```

## Final Notes
This framework is modular and extensible:
- Swap out `UCB` for other acquisition functions (e.g., `EI`, `PI`).
- Replace `Nystroem kernel` with neural embeddings.
- Integrate real-time health estimation models for better early prediction.

It has been successfully deployed on both NEWARE and adapted for LHCE, demonstrating robustness across platforms.

## References
If you find this framework useful, please cite:
```
[Paper Citation Here]
"Bayesian optimization of charging protocol and its mechanism for anode free lithium metal batteries"
Authors, Journal, Year 
```

## Acknowledgments
This work was inspired by and partially builds upon the methods developed by Attia et al. in their fast-charging optimization framework.

Original code: https://github.com/chueh-ermon/battery-fast-charging-optimization

We thank the Ermon Group at Stanford University for open-sourcing their implementation.
