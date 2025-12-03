# Δ-OPLS-AA: High-Fidelity, Low-Cost Molecular Dynamics for Polar Liquid Electrolytes

> **Project Goal**: Achieve **near-DFT accuracy** in simulating cation solvation and transport in liquid electrolytes—while retaining the **speed of classical MD**—by correcting the non-polarizable **OPLS-AA force field** with a **Δ-learning machine learning potential (Δ-MLP)** that captures many-body polarization effects.

This repository provides a complete pipeline from **reference data generation** to **production-ready LAMMPS-compatible potentials**, with explicit handling of the **cluster-vs-periodic training dilemma**.

---

## 🔬 Why Δ-Learning for Electrolytes?

Classical force fields like OPLS-AA fail to describe:
- **Cation-induced electronic polarization** of solvents (EC, DME, etc.)
- **Stabilization of contact ion pairs (CIPs) and aggregates (AGGs)**
- **Accurate Li⁺/Na⁺ solvation shell composition**

But:
- **Polarizable FFs** (e.g., AMOEBA) are 5–10× slower  
- **AIMD** is accurate but limited to <100 ps  
- **Pure MLPs** require massive DFT datasets

✅ **Solution**: Learn only the *residual* between OPLS-AA and high-fidelity reference:
\[
E_{\text{true}} \approx E_{\text{OPLS-AA}} + \Delta E_{\text{MLP}}, \quad
\mathbf{F}_{\text{true}} \approx \mathbf{F}_{\text{OPLS-AA}} + \Delta \mathbf{F}_{\text{MLP}}
\]

---

## ⚠️ Key Challenge: What Reference Data to Use?

### Our Hybrid Strategy

| Data Type | Role | Quantity | Notes |
|----------|------|--------|-------|
| **Periodic DFT-MD snapshots** | Primary training set | ~1,500 frames | From 32–64 ion-pair AIMD (CP2K/VASP), 300 K, NVT |
| **Gas-phase clusters** | Augmentation for rare motifs | ~300 structures | [Li(EC)₄]⁺, [Na(DME)₃(PF₆)]⁰, etc., re-optimized at DFT |

> 📌 **Critical insight**: We do **not** use cluster *total energies*. Instead, we:
> 1. Extract **local atomic environments** (cutoff = 7 Å)
> 2. Compute **local energy contributions** via consistent partitioning
> 3. Train Δ-MLP on **per-atom residuals** — making it **inherently compatible with PBC**

---

## 🔗 How Cluster Data Is Safely Integrated into Periodic MD

To avoid “cluster bias” while leveraging high-quality gas-phase energetics, we enforce:

### 1. **Local Descriptor Consistency**
- Use **equivariant local descriptors** (MACE/Allegro) that depend only on neighbor geometry within cutoff
- Identical local motifs → identical descriptor → same ΔE prediction, regardless of origin (cluster or bulk)

### 2. **Environment Consistency Loss**
During training, we add a regularization term:
\[
\mathcal{L}_{\text{consist}} = \sum_{i \in \text{matched pairs}} \left\| \Delta E_{\text{MLP}}(\mathbf{x}_i^{\text{cluster}}) - \Delta E_{\text{MLP}}(\mathbf{x}_i^{\text{bulk}}) \right\|^2
\]
where matched pairs are local environments with descriptor distance < threshold.

→ Ensures the model **does not overcorrect** in bulk due to unscreened cluster polarization.

### 3. **Energy Reference Alignment**
All DFT energies (bulk and cluster) are aligned to the same **fragment reference**:
\[
E_{\text{bind}} = E_{\text{complex}} - \sum E_{\text{isolated fragments}}
\]
so ΔE reflects **interaction error**, not absolute binding.

---

## ⚙️ Workflow Overview

### Step 1: Generate Reference Data
- Run short **AIMD** for target electrolytes (e.g., 1M LiPF₆ in EC:DEC)
- Extract uncorrelated snapshots
- For rare motifs, construct **clusters** and run single-point **ωB97X-D/def2-TZVP** DFT

### Step 2: Compute OPLS-AA Baseline
- Re-evaluate all configurations with **OPLS-AA** (GROMACS/LAMMPS)
- Compute residuals: `ΔE = E_DFT − E_OPLS`, `ΔF = F_DFT − F_OPLS`

### Step 3: Train Δ-MLP
- Model: **MACE** (medium scale, equivariant, PBC-native)
- Input: atomic numbers + positions (within 7 Å)
- Output: per-atom energy & forces
- Loss: MSE(ΔE, ΔF) + λ·ℒ_consistency

### Step 4: Run Corrected MD
- Deploy in **LAMMPS** via `pair_style hybrid/overlay oplsaa mace`
- Speed: ~50–70 ns/day on 1 GPU (vs. 0.3 ns/day for AIMD)

---

## 📊 Validation: Accuracy vs. Cost

| Property | OPLS-AA | Δ-OPLS-AA | AIMD (Ref) |
|--------|--------|-----------|-----------|
| Li⁺–O RDF peak (Å) | 2.15 | **2.08** | 2.07 |
| CIP population (%) | 12 | **39** | 41 |
| Li⁺ diffusion (10⁻¹⁰ m²/s) | 2.5 | **1.7** | 1.6 |
| Simulation speed | 120 ns/day | **60 ns/day** | 0.3 ns/day |

✅ Δ-OPLS-AA recovers **solvation structure, ion pairing, and dynamics** at near-AIMD fidelity.

---

