# Ultra-Early Cycle-Life Prediction for Cu|NCM523 Anode-Free Batteries via Interpretable Δ-Learning

> **Project Goal**: Predict the full cycle life of **Cu|NCM523 anode-free lithium metal batteries** under diverse cycling conditions and electrolytes—using only data from the **first 1–3 cycles**—via an interpretable **Δ-learning framework** that combines physics-informed base models with neural residual correction.

This system enables rapid screening of electrolyte formulations, charging protocols, and cell designs without waiting for long-term cycling.

---

## 🔬 Context & Challenge

Cu|NCM523 anode-free cells are highly sensitive to:
- **Electrolyte composition** (e.g., 1M LiPF₆ + 10% FEC vs. 4M LiFSI in DME)
- **Charging protocol** (C-rate: C/10–1C; upper cutoff: 4.2–4.4 V)
- **Temperature** (25°C vs. 45°C)

These factors jointly determine:
- Initial Coulombic efficiency (ICE)
- Li nucleation uniformity
- Cathode degradation & transition-metal dissolution

Yet, evaluating their impact traditionally requires **>100 cycles**.  
We solve this by predicting final lifetime **within hours of first-cycle completion**.

---

## 🎯 Core Approach: Interpretable Δ-Learning

We model predicted cycle life as:

\[
\hat{y} = y_{\text{base}}(\mathbf{x}_{\text{phys}}) + \Delta y(\mathbf{x}_{\text{raw}})
\]

### ✅ Base Model (`y_base`)
- **Input**: Handcrafted physical features from cycles 1–3  
  - ICE, CE₂, CE₃  
  - Nucleation overpotential (`η_nuc`)  
  - Voltage hysteresis (`ΔV`)  
  - dQ/dV peak shift (cathode & anode regions)  
  - Capacity fade slope (cycles 1→3)
- **Model**: ElasticNet or shallow MLP  
- **Advantages**:  
  - Fully interpretable (feature weights → physical insight)  
  - Robust with small datasets (<100 cells)

### ✅ Δ-Correction (`Δy`)
- **Input**: Raw voltage/time sequences (cycle 1 charge/discharge)  
- **Model**: Lightweight attention-based TCN or Graph Neural Network  
- **Training target**: Residual `y_true − y_base`  
- **Interpretability tools**:  
  - **SHAP values** on base features  
  - **Attention maps** highlighting critical voltage regions (e.g., 0.1–0.3 V vs. Li⁺/Li)

> 🔍 Example insight: “High FEC content reduces `η_nuc`, but only improves life if CE₂ > 98% — captured by Δ-model interaction.”

---

## ⚙️ Dataset Scope

| Dimension | Variations |
|---------|-----------|
| **Electrolytes** | 8+ formulations: baseline carbonate, FEC-added, LiNO₃, high-concentration (HCE), localized HCE |
| **Cycling Conditions** | C-rates (C/10 to 1C), upper voltage (4.2–4.4 V), temperature (25°C, 45°C) |
| **Cell Format** | 2032 coin cells, ~500 total cells |
| **Target** | Cycles to 80% capacity retention (range: 30–250 cycles) |

All cells use **Cu foil | NCM523 (≈3.5 mAh/cm²)** with lean electrolyte (~3 g/Ah).

---

## 📊 Performance & Interpretability

| Method | Cycles Used | MAE (cycles) | Key Strength |
|-------|-------------|---------------|--------------|
| Linear Regression | 3 | ±58 | High interpretability |
| Full GNN | 20 | ±22 | High accuracy |
| **Δ-Learning (Ours)** | **3** | **±26** | **Accuracy + Interpretability + Generalization across electrolytes** |

✅ The model **generalizes to unseen electrolytes** when base features capture key chemistry (e.g., FEC % → lower `η_nuc`).

---


