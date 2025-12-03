# High-Throughput Discovery of Lithium-Metal Seed Layers via Universal Machine Learning Potentials

> **Project Goal**: Accelerate the discovery of inorganic seed layers that enable uniform, dendrite-free Li-metal deposition by combining **universal machine learning potentials (UMA/MACE)** with multi-objective high-throughput screening and experimental validation.

This repository integrates **large-scale atomistic simulation**, **materials informatics**, and **electrochemical testing** to identify optimal seed-layer materials that simultaneously satisfy:
1. Electrochemical stability against Li metal,
2. Low lattice mismatch with Li(110),
3. Fast Li surface diffusion kinetics.

---

## 🔬 Scientific Motivation

Lithium metal anodes suffer from uncontrolled dendrite growth due to:
- Non-uniform nucleation
- Unstable solid electrolyte interphase (SEI)
- High interfacial energy

Introducing an artificial **seed layer** on Cu or current collector can:
- Lower nucleation overpotential
- Guide epitaxial Li(110) deposition
- Suppress side reactions

However, brute-force experimental screening of inorganic candidates (oxides, nitrides, sulfides, etc.) is slow. We address this via **physics-informed ML-driven computation**.

---

## 🎯 Screening Criteria

Each candidate material is evaluated on three key descriptors:

| Criterion | Computational Method | Target |
|---------|----------------------|--------|
| **1. Electrochemical Stability** | Ab initio thermodynamics + UMA-based grand-canonical MD | ΔG < 0 for `Material + xLi → LiₓMaterial` (no decomposition) |
| **2. Lattice Mismatch with Li(110)** | Surface lattice matching (using relaxed slabs) | Mismatch < 8% |
| **3. Li Surface Diffusion Barrier** | Nudged Elastic Band (NEB) with UMA potential | Eₐ < 0.15 eV |

Only materials passing **all three filters** proceed to ranking and experimental validation.

---

## ⚙️ Workflow Overview

### Step 1: Candidate Pool Construction
- Start from **Materials Project / OQMD** database (~10,000 inorganic compounds)
- Filter by: non-toxic, air-stable precursors, scalable synthesis
- Final pool: ~500 binary/ternary compounds (e.g., Li₃N, MgF₂, AlN, BN, TiC)

### Step 2: Structure Relaxation & Surface Modeling
- Use **universal ML potential (UMA)** to relax bulk and (001)/(111) surfaces
- Validate UMA against DFT on 50 reference systems (RMSE < 20 meV/atom)

### Step 3: High-Throughput Property Calculation
For each candidate:
1. Compute **formation energy vs. Li** → assess stability
2. Extract **surface lattice parameters** → calculate mismatch with Li(110) (a = 3.51 Å)
3. Run **NEB with UMA** to obtain Li diffusion barrier on surface

### Step 4: Multi-Objective Ranking
- Normalize scores: `S = w₁·Stability + w₂·(1−Mismatch) + w₃·(1−Eₐ)`
- Top 10 candidates selected for synthesis

### Step 5: Experimental Validation
- **Synthesis**: Sputtering / ALD / solution process of seed layer on Cu foil
- **Characterization**: XRD, XPS, SEM
- **Electrochemical Test**:
  - Li|Cu half-cell: nucleation overpotential, CE
  - Li|Li symmetric cell: cycling stability at 1 mA/cm²
  - Full cell (NMC811|Li): rate capability & cycle life

---

## 🤖 Technical Stack

| Component | Tool |
|--------|------|
| Universal ML Potential | UMA or MACE |
| DFT Reference Data | VASP + Pymatgen |
| High-Throughput Engine | AiiDA or custodian + FireWorks |
| NEB Calculations | ASE + UMA interface |
| Data Analysis | pandas, scikit-learn, plotly |
| Visualization | OVITO, pymatgen.analysis.diffusion |

---


