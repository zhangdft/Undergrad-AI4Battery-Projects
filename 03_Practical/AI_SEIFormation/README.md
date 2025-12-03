# AI-Driven SEI Pre-Formation Optimization for Anode-Free Lithium Metal Batteries

> **Project Goal**: Use active learning to discover optimal *formation protocols* that program high-quality solid electrolyte interphases (SEI) on bare Cu current collectors—enabling stable, high-efficiency anode-free Li metal batteries.

This repository implements a **sample-efficient, interpretable AI framework** that combines electrochemical testing with **Random Forest–based active learning** to optimize multi-stage charging protocols (CC/CV/Pulse) for SEI engineering—without requiring in-situ characterization.

---

## 🔬 Problem Context

In anode-free lithium metal batteries (AFLMBs), the initial SEI formed on Cu during the first charge critically determines:
- Coulombic efficiency (CE)
- Li deposition uniformity
- Cycle life and safety

However, the SEI is highly sensitive to the **formation protocol**:
- Constant Current (CC)
- Constant Voltage (CV)
- Pulse charging (on/off)

Manually screening all combinations is infeasible. We treat protocol design as a **sequential decision + active learning problem**.

---

## 🎯 Key Objectives

1. **Optimize formation protocol** (up to 3 stages, each ∈ {CC, CV, PULSE}) to maximize SEI quality.
2. **Use only standard Li|Cu coin cell tests** — no three-electrode or in-situ tools.
3. **Evaluate performance via 10-cycle average CE and its standard deviation** (proxy for SEI stability).
4. **Employ Random Forest + prediction-variance active learning** (distinct from prior BO/GPR work).
5. **Provide white-box interpretation** → link AI recommendations to physical mechanisms via post-hoc characterization.

---

## 🧪 Experimental Protocol

### Battery Configuration
- **Cell type**: Li|Cu coin cell (CR2032)
- **Electrolyte**: User-defined (e.g., 1M LiPF₆ in EC:DEC + 10% FEC)
- **Formation**: One-time protocol per cell (see input space below)
- **Post-formation cycling**: 10 cycles at fixed C-rate (e.g., 0.5 mA/cm²)
- **Metrics extracted**:
  - `CE_avg`: Average CE over cycles 2–10
  - `CE_std`: Standard deviation of CE (lower = more stable SEI)

> ⚠️ **Each experimental round tests ~50 unique protocols**.

---

## 🤖 AI Framework Overview

### Input Space: Formation Protocol Encoding
Each protocol is encoded as a **fixed-length vector** (3 stages max), with unused stages padded:

| Stage | Mode (one-hot) | Parameter(s) |
|-------|----------------|--------------|
| 1     | CC / CV / PULSE | CC: `I (mA/cm²)`, `t (min)`<br>CV: `V (V)`, `t_max (min)`<br>PULSE: `I_on`, `t_on`, `t_off`, `total_t` |
| 2     | ...            | ...          |
| 3     | ...            | ...          |

→ Total input dimension: **~15–20 features** (after one-hot + normalization).

### Output Space (Targets)
- `y₁ = CE_avg`  (↑ higher is better)
- `y₂ = -CE_std` (↑ lower std is better → negate for maximization)

Both normalized to [0, 1].

### Active Learning Loop
1. **Initial pool**: 50 randomly sampled protocols → tested experimentally.
2. **Train multi-output Random Forest** regressor.
3. **Generate candidate pool**: 2,000 virtual protocols (covering realistic CC/CV/PULSE combinations).
4. **Predict mean & variance** for each candidate using tree-wise predictions.
5. **Safety filter**: Remove candidates with predicted `CE_avg < 0.98`.
6. **Select top-5 candidates** with highest **weighted prediction variance**:
   ```python
   score = w1 * Var(CE_avg) + w2 * Var(-CE_std)


- Test selected protocols, add to training set.
- Repeat for 3–5 rounds (~200 total samples).

### Why Random Forest?

- Handles mixed discrete/continuous inputs naturally.
- Robust with n ≈ 200 samples (outperforms GPR when response is non-smooth).
- Enables uncertainty quantification via inter-tree variance.
- Fast training/inference → ideal for lab deployment.

---

🔍 Interpretability & Validation

After convergence, we perform two levels of explanation:

1. Algorithmic White-Box Analysis

- **Feature importance**: Built-in RF Gini importance.
- **SHAP values**: Quantify how each protocol parameter (e.g., "CV hold time") influences CE.
- **Partial dependence plots**: Reveal nonlinear effects (e.g., "CE peaks at 20-min CV").

2. Physical Validation via Ex-Situ Characterization

Top 3 AI-recommended vs. baseline (e.g., standard CC) protocols are validated with:

- **XPS**: Quantify SEI composition (LiF, Li₂CO₃, ROLi ratios)
- **SEM**: Assess Li morphology after 10 cycles
- **ToF-SIMS**: Depth profiling of inorganic/organic layers

→ Bridge AI recommendation ↔ SEI chemistry/morphology.
