# Unraveling Fast Ion Transport in High-Entropy Sulfide Solid Electrolytes via Machine Learning Potentials

> **Project Goal**: Decipher the microscopic origins of enhanced Li⁺ conductivity in high-entropy sulfide solid electrolytes by constructing accurate **machine learning potentials (MLPs/ACE)** and performing large-scale molecular dynamics simulations to analyze percolation networks, structural disorder, and dynamic ion pathways.

This project bridges **materials design**, **machine learning**, and **ion transport physics** to establish structure–dynamics–conductivity relationships in entropy-stabilized Li-ion conductors.

---

## 🔬 Scientific Background

High-entropy sulfides (e.g., Li₆PS₅X with X = Cl/Br/I/O or multi-cation variants like Li₆P₁₋ₓMₓS₅Cl, M = Ge/Sn/Sb/Bi) exhibit:
- Room-temperature ionic conductivity > 10 mS/cm
- Enhanced stability vs. moisture or Li metal
- Suppressed phase segregation

But the **mechanism behind fast ion transport** remains unclear:
- Is it due to **lattice softening**?
- **Percolating low-energy pathways**?
- **Dynamic site disorder** induced by cation mixing?

Conventional DFT-based MD is too costly for the required system size (>500 atoms) and simulation time (>1 ns). We solve this with **machine learning potentials**.

---

## 🎯 Objectives

1. **Construct high-fidelity MLPs** (using ACE or Allegro/NequIP) for Li–P–S–(M₁,M₂,…,Mₙ) systems.
2. **Validate MLPs** against DFT on energies, forces, and phonon spectra.
3. **Perform MLP-driven MD** (≥1 ns, ≥1000 atoms) to compute Li⁺ diffusivity and conductivity.
4. **Quantify key microstructural features**:
   - **Percolation network** of Li sites (via graph theory)
   - **Site energy disorder** (from local coordination environments)
   - **Dynamic bottleneck fluctuations** (time-resolved migration barriers)
5. **Correlate entropy-induced disorder with ion mobility**.

---

## ⚙️ Methodology Workflow

### 1. Dataset Generation
- Generate diverse structures: random cation substitutions, Li-vacancy configurations, amorphous phases
- Run **DFT calculations** (VASP) → collect energies & atomic forces (~5,000 configurations)

### 2. Machine Learning Potential Training
- Use **Atomic Cluster Expansion (ACE)** or **equivariant MLPs** (e.g., Allegro)
- Target: < 20 meV/atom energy error, < 0.1 eV/Å force error
- Include long-range electrostatics via Ewald or machine-learned charges

### 3. Large-Scale MLP-MD Simulations
- System size: 512–2048 atoms
- Simulation time: 1–5 ns at 300–600 K
- Compute:
  - Li⁺ diffusion coefficient (`D_Li`) via MSD
  - Ionic conductivity (`σ`) via Nernst-Einstein relation
  - Residence times, jump frequencies

### 4. Microstructural Analysis
- **Percolation analysis**: Build Li-site graph; identify connected pathways above percolation threshold
- **Local environment fingerprints**: Use SOAP or ACE descriptors to classify Li sites
- **Energy landscape mapping**: Project Li migration events onto local structural motifs
- **Dynamic bottleneck tracking**: Monitor S–S or M–S bond fluctuations during Li hops

---

## 🤖 Technical Stack

| Component | Tool |
|--------|------|
| DFT Data Generation | VASP + pymatgen + custodian |
| MLP Framework | [ACEsuit](https://github.com/ACEsuit/ace) / [Allegro](https://github.com/mir-group/allegro) |
| MD Engine | LAMMPS (with ACE plugin) or JAX-MD |
| Analysis | MDAnalysis, freud, networkx, scikit-learn |
| Visualization | OVITO, VESTA, plotly |

---


