# AI-Driven Discovery and Validation of ADRB2 Ligands

**A computational pipeline combining Machine Learning and Molecular Docking for drug discovery targeting the Beta-2 Adrenergic Receptor (ADRB2)**

---

## 🎯 Project Overview

This project implements a complete computational drug discovery workflow for identifying potential therapeutic candidates for respiratory conditions like asthma. The pipeline combines:

- **AI/Machine Learning**: Random Forest models for rapid virtual screening
- **Physics-Based Validation**: AutoDock Vina for structural docking
- **Data-Driven Approach**: ChEMBL bioactivity database integration

### Key Features

✅ Automated data acquisition from ChEMBL database  
✅ Morgan fingerprint-based molecular featurization  
✅ Random Forest regression for activity prediction  
✅ Virtual screening of candidate libraries  
✅ AutoDock Vina integration for 3D docking  
✅ Comprehensive visualization and reporting  

---

## 📁 Project Structure

```
adrb2_discovery/
├── data/                          # Training and candidate data
│   ├── chembl_adrb2_data.csv     # ChEMBL bioactivity data
│   └── potential_candidates.csv   # Virtual screening library
├── models/                        # Trained ML models
│   └── adrb2_rf_model.pkl        # Random Forest model
├── protein/                       # Receptor structures
│   └── beta2_receptor.pdb        # ADRB2 3D structure
├── docking/                       # Docking workflow
│   ├── receptor.pdbqt            # Prepared receptor
│   ├── vina_config.txt           # Docking configuration
│   ├── ligands/                  # Prepared ligand files
│   └── results/                  # Docking outputs
├── results/                       # Analysis results
│   ├── ai_predictions.csv        # Virtual screening results
│   ├── final_hits.csv            # Combined AI + docking results
│   ├── final_report.txt          # Summary report
│   └── figures/                  # Visualizations
└── scripts/                       # Pipeline scripts
    ├── fetch_data.py             # ChEMBL data downloader
    ├── main_pipeline.py          # AI training & screening
    ├── run_docking.py            # Docking automation
    └── visualize_results.py      # Results analysis
```

---

## 🚀 Quick Start

### Prerequisites

**Required Software:**
- Python 3.8+
- AutoDock Vina (for molecular docking)
- OpenBabel (for file format conversion)

**Install System Dependencies (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install autodock-vina openbabel
```

**Install Python Dependencies:**
```bash
cd adrb2_discovery
pip install -r requirements.txt --break-system-packages
```

---

## 📊 Usage Guide

### Step 1: Data Acquisition

Fetch ADRB2 bioactivity data from ChEMBL database:

```bash
python scripts/fetch_data.py
```

**What it does:**
- Downloads bioactivity data for ADRB2 (UniProt: P07550)
- Filters for molecules with pIC50/pChEMBL values
- Creates example candidate library
- Saves to `data/chembl_adrb2_data.csv`

**Manual Alternative:**
Visit [ChEMBL](https://www.ebi.ac.uk/chembl/target_report_card/P07550/) and download bioactivity data manually.

---

### Step 2: AI Model Training & Virtual Screening

Train the Random Forest model and screen candidates:

```bash
python scripts/main_pipeline.py
```

**Pipeline Stages:**

1. **Data Loading**: Reads ChEMBL bioactivity data
2. **Featurization**: Converts SMILES to Morgan fingerprints (2048-bit)
3. **Model Training**: 
   - Random Forest with 100 estimators
   - 80/20 train-test split
   - Expected R² > 0.70
4. **Virtual Screening**: Predicts binding affinity for candidate library
5. **Molecular Filtering**: Calculates drug-likeness properties
6. **Output**: Top 100 candidates saved to `results/ai_predictions.csv`

**Expected Output:**
```
=== Model Performance ===
Training R²: 0.8523
Test R²: 0.7412
Training RMSE: 0.4231
Test RMSE: 0.5789

=== Virtual Screening Complete ===
Top 100 candidates saved to results/ai_predictions.csv
```

---

### Step 3: Molecular Docking Validation

**Prepare Receptor:**

Place your ADRB2 structure in `protein/beta2_receptor.pdb`. You can obtain this from:
- [RCSB PDB](https://www.rcsb.org/) (search for ADRB2 structures)
- AlphaFold database
- Homology modeling tools

**Configure Docking Parameters:**

Edit `scripts/run_docking.py` to set the binding site coordinates:

```python
config = vina.create_config_file(
    center_x=12.5,   # Update based on your binding site
    center_y=8.3,    # Use PyMOL or Chimera to identify
    center_z=15.7,   # the active site center
    size_x=25.0,     # Search box dimensions
    size_y=25.0,
    size_z=25.0
)
```

**Run Docking:**

```bash
python scripts/run_docking.py
```

**What it does:**
1. Converts receptor PDB → PDBQT format
2. Prepares top 10 ligands from AI predictions
3. Runs AutoDock Vina for each ligand
4. Parses binding affinities
5. Combines AI + docking scores
6. Saves ranked hits to `results/final_hits.csv`

**Expected Output:**
```
=== Docking Complete ===
Successfully docked: 10/10 ligands

=== Final Analysis ===
Combined 10 molecules with both AI and docking scores

Top 5 candidates:
molecule_id  predicted_pIC50  binding_affinity  composite_score
  CAND_0042            7.234             -8.9            0.923
  CAND_0127            6.891             -8.2            0.857
```

---

### Step 4: Visualization & Reporting

Generate comprehensive analysis:

```bash
python scripts/visualize_results.py
```

**Generated Outputs:**

1. **model_performance.png**: R² and RMSE metrics
2. **predictions_analysis.png**: Activity distribution and molecular properties
3. **docking_analysis.png**: AI vs docking correlation
4. **final_report.txt**: Detailed candidate summary

---

## 🧬 Scientific Background

### Beta-2 Adrenergic Receptor (ADRB2)

**Biological Role:**
- G protein-coupled receptor (GPCR)
- Primary target for asthma/COPD treatments
- Mediates smooth muscle relaxation in airways

**Current Therapeutics:**
- Albuterol (short-acting agonist)
- Salmeterol (long-acting agonist)
- Target for next-generation bronchodilators

### Machine Learning Approach

**Why Random Forest?**
- Handles high-dimensional fingerprint data effectively
- Non-linear relationship modeling
- Built-in feature importance
- Robust to overfitting with proper tuning

**Morgan Fingerprints:**
- Circular topological representation
- Radius = 2 captures structural motifs
- 2048-bit vector balances information vs. sparsity
- Industry standard for QSAR modeling

### Molecular Docking

**AutoDock Vina:**
- Physics-based scoring function
- Accounts for:
  - Hydrogen bonding
  - Hydrophobic interactions
  - Steric clashes
  - Entropic penalties
- Scores in kcal/mol (more negative = stronger binding)

**Interpretation:**
- **< -7.0 kcal/mol**: Strong binding (nanomolar range)
- **-5.0 to -7.0**: Moderate binding (micromolar)
- **> -5.0**: Weak or non-binder

---

## 📈 Results Interpretation

### Strong Hit Criteria

A high-quality lead candidate should meet:

✅ **AI Score**: pIC50 > 6.0 (corresponds to IC50 < 1 µM)  
✅ **Docking Score**: < -7.0 kcal/mol  
✅ **Composite Score**: > 0.7  
✅ **Drug-likeness**: Passes Lipinski's Rule of Five  

### Lipinski's Rule of Five

For oral bioavailability:
- Molecular Weight ≤ 500 Da
- LogP ≤ 5
- H-bond Donors ≤ 5
- H-bond Acceptors ≤ 10

### Next Steps After Hit Identification

1. **Experimental Validation**:
   - Radioligand binding assays
   - Functional cAMP assays
   - Selectivity screening (ADRB1, ADRB3)

2. **Lead Optimization**:
   - SAR studies (structure-activity relationship)
   - ADME profiling
   - Toxicity assessment

3. **Structural Validation**:
   - X-ray crystallography or Cryo-EM
   - Confirm binding pose

---

## 🔧 Customization

### Using Different Targets

Modify `fetch_data.py`:
```python
training_data = fetch_chembl_data(
    target_uniprot='YOUR_UNIPROT_ID',  # e.g., 'P35368' for ADRB1
    output_dir='/path/to/data'
)
```

### Adjusting ML Hyperparameters

Edit `main_pipeline.py`:
```python
self.model = RandomForestRegressor(
    n_estimators=200,      # Increase for better performance
    max_depth=30,          # Control tree complexity
    min_samples_split=3,   # Prevent overfitting
    random_state=42
)
```

### Custom Candidate Libraries

Replace `data/potential_candidates.csv` with your library:
```csv
molecule_id,smiles,source
VENDOR_001,CCO,ChemBridge
VENDOR_002,CC(C)O,Enamine
```

---

## ⚠️ Important Notes

### Data Quality
- ChEMBL data heterogeneity: Different assay types, conditions
- Recommendation: Filter by assay_type = 'B' (binding assays)
- Remove duplicates and outliers before training

### Docking Limitations
- **Rigid receptor**: Doesn't account for protein flexibility
- **Scoring function accuracy**: ~2-3 kcal/mol error typical
- **False positives**: ~30-50% of computational hits fail in vitro
- **Best use**: Enrichment tool, not absolute predictor

### Computational Resources
- Model training: ~5-10 minutes (1000 molecules)
- Virtual screening: ~1 second per molecule
- Docking: ~2-5 minutes per ligand
- Recommend: ≥8GB RAM, multicore CPU

---

## 📚 References

### Key Papers

1. **Machine Learning in Drug Discovery**:
   - Chen et al., "The rise of deep learning in drug discovery" (2018)
   - Vamathevan et al., "Applications of machine learning in drug discovery" (2019)

2. **AutoDock Vina**:
   - Trott & Olson, "AutoDock Vina: improving the speed and accuracy" (2010)
   - Eberhardt et al., "AutoDock Vina 1.2.0" (2021)

3. **ADRB2 Biology**:
   - Rasmussen et al., "Crystal structure of the β2 adrenergic receptor" (2007)
   - Johnson, "Update on β2-agonists in asthma management" (2020)

### Databases

- [ChEMBL](https://www.ebi.ac.uk/chembl/): Bioactivity database
- [PDB](https://www.rcsb.org/): Protein structures
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/): Chemical information
- [DrugBank](https://www.drugbank.com/): Drug information

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Deep learning models (Graph Neural Networks)
- [ ] Pharmacophore filtering
- [ ] ADMET prediction integration
- [ ] Multi-objective optimization
- [ ] Ensemble docking methods
- [ ] Fragment-based screening

---

## 📄 License

This project is for educational and research purposes. Ensure compliance with:
- ChEMBL license terms
- AutoDock Vina GPL license
- Your institution's computational resources policy

---

## 📞 Support

For issues or questions:
1. Check existing documentation
2. Review scientific literature
3. Consult with medicinal chemistry experts
4. File issue with detailed error logs

---

## 🎓 Educational Use

This pipeline is designed for:
- Computational chemistry students
- Drug discovery researchers
- Bioinformatics training
- Academic projects

**Not suitable for**:
- Clinical decision making
- Direct drug development without validation
- Unsupervised use by non-experts

---

## ✅ Quality Checklist

Before reporting results:

- [ ] Verified ChEMBL data quality
- [ ] Model R² > 0.65 on test set
- [ ] Checked for SMILES parsing errors
- [ ] Validated docking box placement
- [ ] Reviewed drug-likeness metrics
- [ ] Compared to known ADRB2 ligands
- [ ] Documented all parameters used

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Author**: AI-Driven Drug Discovery Pipeline  

---

*Disclaimer: This is a computational tool for research purposes only. All hits must be validated experimentally before any therapeutic application.*
