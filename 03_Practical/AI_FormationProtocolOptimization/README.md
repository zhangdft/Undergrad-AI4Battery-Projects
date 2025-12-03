# AI-Driven Formation Protocol Optimization for Long-Life Lithium Metal Batteries

> **Project Goal**: Discover optimal *formation protocols* that engineer stable electrode-electrolyte interfaces in Li-metal batteries, thereby maximizing cycle life—guided by an AI model that predicts lifetime from early-cycle data.

This repository implements an active learning system that discovers formation protocols maximizing predicted battery lifetime, based on short-term testing and a pre-trained lifetime estimator.

---

## 🔬 Problem Context

In practical Li-metal batteries (e.g., NMC|Li), the initial formation process critically determines:
- SEI/CEI stability
- Coulombic efficiency in early cycles
- Long-term capacity fade rate

However, traditional trial-and-error optimization is prohibitively slow due to the need for **hundreds of cycles** to assess lifetime.

We accelerate this by:
- Defining a **structured two-stage formation protocol**
- Running only **short-term tests** (~50 cycles total)
- Using an **AI lifetime predictor** to estimate final cycle life
- Feeding this prediction into an **active learning loop**

---

## 🎯 Key Objectives

1. **Optimize two-stage formation protocol**:
   - Stage 1: Two-step charge to 100% SOC (each step ∈ {CC, CV, PULSE})
   - Stage 2: `n` cycles of constant-current charge/discharge at fixed capacity (e.g., 1 mAh/cm²)
2. **Use AI-predicted cycle life** (to 80% capacity retention) as performance metric.
3. **Employ Random Forest + uncertainty-based active learning** (sample-efficient, <200 experiments).
4. **Provide interpretable insights**: which formation parameters most extend life?
5. **Validate top candidates with full long-cycle testing**.

---

## ⚙️ Formation Protocol Definition

Each protocol is fully defined by:

### Stage 1: Two-Step Charge to Full
| Step | Mode       | Parameters |
|------|------------|-----------|
| 1    | CC / CV / PULSE | e.g., CC: `I₁`, `Q₁` or `t₁`; CV: `V₁`, `t_max₁`; PULSE: `I_on₁`, `t_on₁`, `t_off₁`, `Q_total₁` |
| 2    | CC / CV / PULSE | Same parameterization, continues to 100% SOC |

> Total charge must reach full capacity (verified by voltage plateau or coulombic count).

### Stage 2: Stabilization Cycling
- Perform `n` cycles (e.g., `n = 3` or `5`)
- Fixed current density (e.g., 0.5 mA/cm²)
- Fixed areal capacity (e.g., 1.0 mAh/cm²)
- No voltage cutoff override

→ This mimics industrial “formation + aging” steps.

---

## 🧪 Experimental Workflow

1. Fabricate Li-metal full cells (e.g., NMC811|Li).
2. Apply one candidate formation protocol (as defined above).
3. Continue cycling at C/2 for **50 additional cycles** (total ~55 cycles).
4. Extract early-cycle features:
   - Capacity retention (cycles 1–10, 10–50)
   - dQ/dV peak shifts
   - CE trends
   - Impedance growth (if available)
5. Feed features into **pre-trained lifetime prediction model** → output: `predicted_cycles_to_80pct`.

This prediction becomes the **optimization target**.

---

## 🤖 AI Framework

### Input Space
- Encoded formation protocol: ~20–25 features
  - One-hot mode for each of 2 charge steps
  - Numerical parameters (current, time, capacity, etc.)
  - `n` (stage-2 cycle count) as discrete variable

### Output
- Single scalar: `predicted_cycle_life` (higher = better)

### Active Learning Loop
1. Start with 50 random protocols → test → predict lifetime.
2. Train **multi-output Random Forest** (for mean + variance).
3. Generate 2,000 virtual protocols within bounds.
4. Filter unsafe candidates (e.g., predicted CE < 0.97).
5. Select top-5 by **prediction variance** (uncertainty sampling).
6. Test → update dataset → repeat (3–4 rounds, ~200 total samples).

### Why Random Forest?
- Handles mixed discrete/continuous inputs
- Robust at small sample sizes (n ≈ 200)
- Provides uncertainty via inter-tree variance
- Compatible with SHAP for interpretation

---

## 🔍 Interpretability & Validation

### Algorithmic Explanation
- **SHAP values**: Quantify impact of "CV hold time in step 2" on predicted life
- **Feature importance**: Is stage-2 cycle count (`n`) critical?
- **Partial dependence plots**: Reveal optimal pulse frequency

### Physical Validation
Top 3 protocols undergo:
- **Full cycling to failure** (validate AI prediction accuracy)
- **Post-mortem analysis**:
  - SEM: Li dendrite suppression
  - XPS: Inorganic-rich SEI (LiF, Li₃N)
  - Cross-section FIB-SEM: Electrode delamination

→ Connect **formation design → interface quality → lifetime**

---

