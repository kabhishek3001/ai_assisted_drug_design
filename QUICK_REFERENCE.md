# ADRB2 Drug Discovery - Quick Reference

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt --break-system-packages

# 2. Download data
python scripts/fetch_data.py

# 3. Train model & screen
python scripts/main_pipeline.py

# 4. Run docking
python scripts/run_docking.py

# 5. Analyze results
python scripts/visualize_results.py
```

## 📊 Key Thresholds

| Metric | Good | Excellent |
|--------|------|-----------|
| Model R² | >0.70 | >0.80 |
| Predicted pIC50 | >6.0 | >7.0 |
| Docking Score | <-7.0 | <-8.5 |
| Composite Score | >0.70 | >0.85 |

## 🎯 Lipinski's Rule of Five

✅ MW ≤ 500 Da  
✅ LogP ≤ 5  
✅ HBD ≤ 5  
✅ HBA ≤ 10  

## 🔧 Common Commands

### Check Installation
```bash
python -c "import rdkit, sklearn; print('OK')"
vina --version
obabel -V
```

### Data Validation
```bash
# Count molecules
wc -l data/chembl_adrb2_data.csv

# Check columns
head -1 data/chembl_adrb2_data.csv
```

### Quick Model Test
```python
from main_pipeline import ADRB2DrugDiscovery
pipeline = ADRB2DrugDiscovery()
pipeline.load_and_prepare_data()
pipeline.train_model()
```

## 📁 Key Files

```
data/
├── chembl_adrb2_data.csv       # Training data
└── potential_candidates.csv    # Screening library

protein/
└── beta2_receptor.pdb          # Receptor structure

results/
├── ai_predictions.csv          # Virtual screening
├── final_hits.csv              # Top candidates
└── final_report.txt            # Summary

docking/
├── vina_config.txt             # Docking parameters
└── receptor.pdbqt              # Prepared receptor
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: rdkit` | `conda install -c conda-forge rdkit` |
| `vina: command not found` | `sudo apt-get install autodock-vina` |
| Low R² (<0.60) | More data / filter assay types |
| Docking fails | Check receptor path & box coordinates |

## 🎨 Vina Config Template

```
receptor = docking/receptor.pdbqt
center_x = 12.5    # UPDATE
center_y = 8.3     # THESE
center_z = 15.7    # VALUES
size_x = 25.0
size_y = 25.0
size_z = 25.0
exhaustiveness = 8
```

## 📈 Score Interpretation

### AI Predicted pIC50
- **>7.0**: Strong binder (IC50 < 100 nM)
- **6.0-7.0**: Moderate (100 nM - 1 µM)
- **<6.0**: Weak (>1 µM)

### Docking Affinity (kcal/mol)
- **<-8.5**: Very strong
- **-7.0 to -8.5**: Strong
- **-5.0 to -7.0**: Moderate
- **>-5.0**: Weak

### Composite Score
- **0.85-1.0**: Priority candidates
- **0.70-0.85**: Good candidates
- **0.50-0.70**: Borderline
- **<0.50**: Unlikely hits

## 🧪 Next Steps After Hit ID

1. ✅ Synthesize/purchase top 3-5 compounds
2. ✅ Radioligand binding assay (Ki determination)
3. ✅ Functional assay (cAMP, β-arrestin)
4. ✅ Selectivity (ADRB1, ADRB3)
5. ✅ ADME profiling
6. ✅ In vivo efficacy

## 📚 Essential Links

- ChEMBL: https://www.ebi.ac.uk/chembl/
- PDB: https://www.rcsb.org/
- RDKit Docs: https://www.rdkit.org/docs/
- Vina Manual: https://autodock-vina.readthedocs.io/

## 💡 Pro Tips

- **Always** visualize docked poses in PyMOL
- **Cross-validate** with 5-fold CV minimum
- **Benchmark** against known ADRB2 ligands
- **Document** all parameter choices
- **Validate** computationally before synthesis

## ⚡ Performance Optimization

```python
# Parallel processing
from joblib import Parallel, delayed

# GPU acceleration (if available)
import cupy  # For compatible operations

# Batch processing
for batch in chunks(candidates, batch_size=1000):
    process_batch(batch)
```

## 🔒 Data Quality Checklist

- [ ] No missing SMILES
- [ ] No missing pChEMBL values
- [ ] Removed duplicates
- [ ] Filtered by assay_type
- [ ] Checked for outliers
- [ ] Validated SMILES syntax
- [ ] Standardized chemical structures

## 📊 Success Criteria

**Computational:**
- [ ] Model R² > 0.70
- [ ] At least 10 hits with score > 0.70
- [ ] Diverse chemical scaffolds
- [ ] Pass drug-likeness filters

**Experimental:**
- [ ] Ki < 100 nM (at least 1 hit)
- [ ] Functional activity confirmed
- [ ] Selectivity vs ADRB1 >10-fold
- [ ] No cytotoxicity <100 µM

---

**Print this page for lab reference!**

Version 1.0 | February 2026
