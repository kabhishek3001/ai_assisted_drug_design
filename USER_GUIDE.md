# ADRB2 Drug Discovery Pipeline - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Workflow Overview](#workflow-overview)
4. [Step-by-Step Tutorial](#step-by-step-tutorial)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)
7. [FAQ](#faq)

---

## Introduction

This pipeline implements a complete computational drug discovery workflow combining:
- **Machine Learning**: For rapid prediction of binding activity
- **Molecular Docking**: For structural validation
- **Data Integration**: ChEMBL database connectivity

**Target**: Beta-2 Adrenergic Receptor (ADRB2)  
**Application**: Respiratory therapeutics (asthma, COPD)  
**Approach**: AI-augmented structure-based drug design

---

## Installation

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB free space
- OS: Linux (Ubuntu 20.04+), macOS, Windows (WSL2)

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- Storage: 10 GB
- GPU: Not required but can accelerate some operations

### Software Dependencies

#### 1. Python Environment

```bash
# Check Python version (3.8+ required)
python3 --version

# Create virtual environment (recommended)
python3 -m venv adrb2_env
source adrb2_env/bin/activate  # Linux/Mac
# OR
adrb2_env\Scripts\activate  # Windows
```

#### 2. Python Packages

```bash
cd adrb2_discovery
pip install -r requirements.txt --break-system-packages
```

**Key packages installed:**
- `rdkit`: Chemical informatics
- `scikit-learn`: Machine learning
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `meeko`: Docking preparation

#### 3. System Tools

**AutoDock Vina** (molecular docking):
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install autodock-vina

# macOS
brew install autodock-vina

# Windows
# Download from: https://vina.scripps.edu/
# Add to PATH
```

**OpenBabel** (format conversion):
```bash
# Ubuntu/Debian
sudo apt-get install openbabel

# macOS
brew install open-babel

# Windows
# Download from: http://openbabel.org/
```

### Verify Installation

```bash
# Test Python imports
python3 -c "import rdkit; import sklearn; print('✅ Python packages OK')"

# Test system tools
vina --version
obabel -V
```

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   ADRB2 DRUG DISCOVERY WORKFLOW             │
└─────────────────────────────────────────────────────────────┘

Phase 1: Data Acquisition
├── ChEMBL database query (ADRB2 bioactivity)
├── Data cleaning and filtering
└── Candidate library preparation
         ↓
Phase 2: AI Model Training
├── SMILES → Morgan Fingerprints (2048-bit)
├── Random Forest Regressor training
├── Model validation (R² > 0.70)
└── Save trained model
         ↓
Phase 3: Virtual Screening
├── Load candidate molecules
├── Generate fingerprints
├── Predict pIC50 values
├── Calculate molecular properties
└── Rank top 100 candidates
         ↓
Phase 4: Molecular Docking
├── Prepare receptor (PDB → PDBQT)
├── Prepare top 10 ligands (SMILES → 3D → PDBQT)
├── Configure docking box (binding site)
├── Run AutoDock Vina
└── Parse binding affinities
         ↓
Phase 5: Results Integration
├── Combine AI + docking scores
├── Calculate composite score
├── Generate visualizations
├── Create final report
└── Identify top hits for validation
```

**Timeline:**
- Data acquisition: 5-10 minutes
- Model training: 5-15 minutes
- Virtual screening: 1-5 minutes
- Docking (10 ligands): 20-50 minutes
- Analysis: 2-5 minutes

**Total: ~40-90 minutes** for complete pipeline

---

## Step-by-Step Tutorial

### Step 1: Data Acquisition

#### Option A: Automatic Download (Recommended)

```bash
cd adrb2_discovery
python scripts/fetch_data.py
```

This will:
- Query ChEMBL REST API for ADRB2 (P07550)
- Download bioactivity data (IC50, Ki, Kd, EC50)
- Filter for molecules with pChEMBL values
- Save to `data/chembl_adrb2_data.csv`
- Create example candidate library

**Expected Output:**
```
Total activities retrieved: 2,847
Processed dataset size: 1,523 unique molecules
pChEMBL range: 4.22 - 10.11
Mean pChEMBL: 6.45
```

#### Option B: Manual Download

1. Visit [ChEMBL ADRB2 page](https://www.ebi.ac.uk/chembl/target_report_card/P07550/)
2. Navigate to "Bioactivities" tab
3. Apply filters:
   - Standard Type: IC50, Ki, Kd, EC50
   - pChEMBL Value: Not null
4. Download as CSV
5. Save as `data/chembl_adrb2_data.csv`

**Required columns:**
- `canonical_smiles`: SMILES string
- `pchembl_value`: Negative log of activity (higher = better)

#### Preparing Candidate Library

Your candidate molecules should be in CSV format:

```csv
molecule_id,smiles,source
CAND_0001,CCO,vendor_catalog
CAND_0002,CC(C)O,in_house_library
```

**Sources for candidate libraries:**
- ZINC database (free)
- Enamine REAL (commercial)
- ChemBridge (commercial)
- Your own compound collection

**Save as:** `data/potential_candidates.csv`

---

### Step 2: AI Model Training & Virtual Screening

```bash
python scripts/main_pipeline.py
```

#### What happens:

**2.1 Data Loading**
- Reads ChEMBL bioactivity data
- Validates SMILES strings
- Reports dataset statistics

**2.2 Featurization**
- Converts each SMILES to Morgan fingerprint
- Fingerprint parameters:
  - Radius: 2 (covers 4-bond environment)
  - Bits: 2048 (standard size)
  - Type: Binary (presence/absence of substructure)

**2.3 Model Training**
```
Random Forest Configuration:
- Trees: 100
- Max depth: 20
- Min samples split: 5
- Scoring: R² (coefficient of determination)
```

**Expected Performance:**
- Training R²: 0.80-0.90
- Test R²: 0.70-0.80 (target: >0.70)
- RMSE: 0.5-0.7 pIC50 units

**If R² < 0.70:**
1. Check data quality (remove outliers)
2. Increase n_estimators (150-200)
3. Try different fingerprint radius (3-4)
4. Consider ensemble of models

**2.4 Virtual Screening**
- Loads candidate library
- Generates fingerprints
- Predicts pIC50 for each molecule
- Calculates Lipinski properties:
  - Molecular Weight
  - LogP (lipophilicity)
  - H-bond donors/acceptors

**2.5 Output**

`results/ai_predictions.csv` contains:
```
molecule_id,smiles,predicted_pIC50,mol_weight,logp,hbd,hba
CAND_0042,CC(C)NCC(O)c1ccc(O)c(O)c1,7.234,211.26,0.45,4,4
```

Top 100 candidates selected by predicted activity.

---

### Step 3: Receptor Preparation

#### 3.1 Obtain ADRB2 Structure

**Option A: From PDB Database**
1. Visit [RCSB PDB](https://www.rcsb.org/)
2. Search for "ADRB2" or "beta-2 adrenergic"
3. Recommended structures:
   - **2RH1**: ADRB2-carazolol (antagonist)
   - **3SN6**: ADRB2-BI-167107 (agonist)
   - **3P0G**: ADRB2-formoterol (agonist)
4. Download PDB file
5. Save to `protein/beta2_receptor.pdb`

**Option B: AlphaFold**
1. Visit [AlphaFold DB](https://alphafold.ebi.ac.uk/)
2. Search for UniProt P07550
3. Download predicted structure
4. Note: May need refinement for docking

**Option C: Homology Modeling**
- Use SWISS-MODEL or Modeller
- Template: Related GPCR structures
- Validate model quality (Ramachandran plot)

#### 3.2 Structure Preparation

**Clean the structure:**
1. Remove water molecules
2. Remove non-standard residues (if any)
3. Keep only protein chain A
4. Add missing residues (if needed)

**Using PyMOL:**
```python
# In PyMOL
remove solvent
remove hetatm
save protein/beta2_receptor_clean.pdb
```

**Using Chimera:**
```
Tools → Structure Editing → Dock Prep
- Add hydrogens
- Assign charges
- Delete solvent
```

---

### Step 4: Identify Binding Site

This is **critical** for docking accuracy!

#### 4.1 From Co-crystal Structure

If using a structure with bound ligand (e.g., 2RH1):

```python
# PyMOL
select binding_site, chain A within 5 of organic
show surface, binding_site
```

Record the center coordinates:
```python
centerofmass binding_site
# Example output: X=12.5, Y=8.3, Z=15.7
```

#### 4.2 From Literature

ADRB2 binding pocket residues (from literature):
- **TM3**: Asp113, Val114, Val117
- **TM5**: Phe193, Ser203, Ser204, Ser207
- **TM6**: Trp286, Phe289, Phe290
- **TM7**: Asn312, Tyr316

Calculate centroid of these residues.

#### 4.3 Computational Prediction

Use binding site prediction tools:
- **CASTp**: Pocket detection
- **FTMap**: Solvent mapping
- **DoGSiteScorer**: Druggability assessment

#### 4.4 Configure Docking Box

Edit `scripts/run_docking.py`:

```python
config = vina.create_config_file(
    # UPDATE THESE VALUES
    center_x=12.5,    # X coordinate of binding site center
    center_y=8.3,     # Y coordinate
    center_z=15.7,    # Z coordinate
    
    # Box size (usually 20-30 Å)
    size_x=25.0,      # Should encompass entire pocket
    size_y=25.0,
    size_z=25.0,
    
    exhaustiveness=8  # Search thoroughness (8-32)
)
```

**Verify box placement:**
```python
# PyMOL - visualize box
pseudoatom box_center, pos=[12.5, 8.3, 15.7]
show sphere, box_center
```

---

### Step 5: Run Molecular Docking

```bash
python scripts/run_docking.py
```

#### 5.1 Receptor Conversion

```
beta2_receptor.pdb → receptor.pdbqt
```

Adds:
- Polar hydrogens
- Gasteiger charges
- Removes non-polar hydrogens

#### 5.2 Ligand Preparation

For each of top 10 AI-predicted ligands:

```
SMILES → 3D structure (RDKit)
        ↓
3D optimization (MMFF94)
        ↓
PDBQT format (Meeko)
```

Output: `docking/ligands/ligand_XXXX.pdbqt`

#### 5.3 Docking Execution

For each ligand:
```bash
vina --config vina_config.txt \
     --ligand ligand_XXXX.pdbqt \
     --out ligand_XXXX_docked.pdbqt
```

**Vina searches for:**
- Optimal binding pose
- Multiple conformations (modes)
- Binding energy (kcal/mol)

**Per-ligand runtime:** 2-5 minutes

#### 5.4 Results Parsing

Vina output format:
```
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -8.9       0.000      0.000  ← BEST SCORE
   2       -8.2       1.523      2.134
   3       -7.8       2.451      3.687
```

**Interpretation:**
- Mode 1: Best binding affinity
- **< -7.0**: Strong binding (likely active)
- **-5 to -7**: Moderate binding
- **> -5**: Weak/non-binding

---

### Step 6: Results Analysis

```bash
python scripts/visualize_results.py
```

#### Generated Outputs

**1. final_hits.csv**

Combined AI + docking scores:
```
molecule_id,predicted_pIC50,binding_affinity,composite_score
CAND_0042,7.234,-8.9,0.923
CAND_0127,6.891,-8.2,0.857
```

**2. Visualizations**

- `model_performance.png`: Training metrics
- `predictions_analysis.png`: Activity distributions
- `docking_analysis.png`: AI vs docking correlation
- `learning_curve.png`: Model generalization
- `residual_analysis.png`: Prediction errors

**3. Final Report**

`final_report.txt` summarizes:
- Dataset statistics
- Top candidate recommendations
- Drug-likeness assessment
- Next steps for validation

#### Interpreting Composite Score

```
Composite Score = (AI_score_norm + Docking_score_norm) / 2
```

Where:
- AI_score_norm: Normalized predicted pIC50 (0-1)
- Docking_score_norm: Normalized binding affinity (0-1)

**Selection criteria:**
- Composite score > 0.70
- Predicted pIC50 > 6.0
- Binding affinity < -7.0 kcal/mol

**Priority for synthesis:**
1. High composite score (>0.80)
2. Passes Lipinski's Rule of 5
3. Novel scaffold (not known ADRB2 ligand)
4. Synthetic accessibility

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'rdkit'"

**Solution:**
```bash
# Conda (recommended for RDKit)
conda install -c conda-forge rdkit

# OR pip
pip install rdkit --break-system-packages
```

#### 2. "Vina not found"

**Solution:**
```bash
# Check installation
which vina

# If not found, install:
sudo apt-get install autodock-vina

# Verify
vina --version
```

#### 3. Low R² Score (<0.60)

**Possible causes:**
- Noisy data (mixed assay types)
- Insufficient training data
- Poor fingerprint representation

**Solutions:**
```python
# Filter data by assay type
df = df[df['assay_type'] == 'B']  # Binding assays only

# Increase model complexity
n_estimators=200
max_depth=30

# Try different fingerprints
# ECFP6 (radius=3) instead of ECFP4
radius=3
```

#### 4. Docking Failures

**Error: "Could not open receptor.pdbqt"**
- Check file path
- Ensure receptor.pdbqt exists in docking/ directory

**Error: "Grid box too small"**
- Increase box dimensions (size_x, size_y, size_z)
- Minimum recommended: 20 Å

**Error: "All poses have high energy"**
- Check box placement (may not be on binding site)
- Verify receptor preparation

#### 5. Memory Issues

If training crashes with memory error:
```python
# Reduce fingerprint size
n_bits=1024  # Instead of 2048

# Or process in batches
batch_size = 100
```

---

## Best Practices

### Data Quality

✅ **DO:**
- Filter for consistent assay types
- Remove duplicates
- Check SMILES validity
- Inspect outliers (pIC50 > 10 or < 3)

❌ **DON'T:**
- Mix binding and functional assays
- Include molecules with missing values
- Ignore assay conditions (pH, temperature)

### Model Validation

✅ **DO:**
- Use holdout test set (20%)
- Perform cross-validation (5-fold minimum)
- Check learning curves
- Analyze residuals

❌ **DON'T:**
- Overfit to training data
- Ignore bias-variance tradeoff
- Skip model diagnostics

### Docking Setup

✅ **DO:**
- Use high-resolution crystal structures (<2.5 Å)
- Verify binding site with literature
- Run control docking (known ligand)
- Check multiple poses

❌ **DON'T:**
- Use low-quality models without validation
- Blindly trust docking scores
- Ignore protein flexibility
- Skip pose visualization

### Result Interpretation

✅ **DO:**
- Consider both AI and docking scores
- Check drug-likeness properties
- Validate with literature precedent
- Plan experimental validation

❌ **DON'T:**
- Rely solely on computational results
- Ignore synthesis feasibility
- Skip selectivity considerations
- Proceed without expert review

---

## FAQ

### Q1: How many molecules do I need for training?

**A:** Minimum 500, recommended 1000+. More data = better model.

### Q2: Can I use this for other targets?

**A:** Yes! Just change the UniProt ID in `fetch_data.py`:
```python
fetch_chembl_data(target_uniprot='YOUR_TARGET_ID')
```

### Q3: How accurate are the predictions?

**A:** 
- AI predictions: ~70-80% correlation with experimental
- Docking: ~2-3 kcal/mol error typical
- Combined: Better enrichment than either alone

### Q4: What if I don't have a receptor structure?

**A:** Options:
1. Use AlphaFold prediction
2. Build homology model
3. Use ligand-based methods only (skip docking)

### Q5: How do I improve model performance?

**A:**
1. Increase training data
2. Try ensemble models
3. Use different fingerprints (MACCS, RDKit)
4. Hyperparameter tuning
5. Feature selection

### Q6: Can I run this on Windows?

**A:** Yes, via WSL2 (Windows Subsystem for Linux):
```powershell
wsl --install
# Then follow Linux instructions
```

### Q7: How long does the full pipeline take?

**A:** 
- Small scale (1K candidates, 10 docked): ~1 hour
- Medium scale (10K candidates, 50 docked): ~3-5 hours
- Large scale (100K+ candidates): Use HPC cluster

### Q8: What's a good composite score threshold?

**A:** 
- Excellent: >0.80
- Good: 0.70-0.80
- Moderate: 0.60-0.70
- Poor: <0.60

### Q9: How do I visualize docked poses?

**A:** Use PyMOL or Chimera:
```bash
pymol protein/beta2_receptor.pdb \
      docking/results/ligand_0042_docked.pdbqt
```

### Q10: What's next after identifying hits?

**A:**
1. **In vitro validation**: Binding assays, functional assays
2. **SAR studies**: Test analogs
3. **ADMET profiling**: Absorption, distribution, metabolism
4. **Selectivity**: Test against ADRB1, ADRB3
5. **Lead optimization**: Improve potency/selectivity
6. **In vivo**: Animal models

---

## Additional Resources

### Documentation
- RDKit: https://www.rdkit.org/docs/
- Scikit-learn: https://scikit-learn.org/
- AutoDock Vina: https://vina.scripps.edu/

### Databases
- ChEMBL: https://www.ebi.ac.uk/chembl/
- PubChem: https://pubchem.ncbi.nlm.nih.gov/
- ZINC: http://zinc.docking.org/
- PDB: https://www.rcsb.org/

### Learning
- Drug Discovery Today (journal)
- Journal of Computer-Aided Molecular Design
- TeachOpenCADD tutorials

---

**Version:** 1.0  
**Last Updated:** February 2026  
**Support:** Check README.md for contact information
