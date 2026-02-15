# ADRB2 Visualization Guide

## 📁 Files Generated

- **Protein Structure:**
  - `adrb2_2RH1.pdb` - ADRB2 receptor structure from PDB

- **Top Candidate Structures:**
  - `structures/*.pdb` - Individual molecule files (PDB format)
  - `structures/*.sdf` - Individual molecule files (SDF format)

- **PyMOL Scripts:**
  - `visualize_adrb2.pml` - Full protein + ligands visualization
  - `view_molecules.pml` - Simple molecule viewer

---

## 🚀 How to Visualize

### Method 1: Using PyMOL (Recommended)

#### Install PyMOL:
```bash
# Ubuntu/Debian
sudo apt install pymol

# Or use conda
conda install -c conda-forge pymol-open-source
```

#### Load Full Visualization:
```bash
cd ~/adrb2_discovery/visualizations
pymol visualize_adrb2.pml
```

#### Or Load Just Molecules:
```bash
pymol view_molecules.pml
```

### Method 2: Using PyMOL GUI

1. Open PyMOL
2. File → Run Script
3. Select `/home/abhishek30/adrb2_discovery/visualizations/visualize_adrb2.pml`

### Method 3: Manual Loading

In PyMOL command line:
```
load /home/abhishek30/adrb2_discovery/visualizations/structures/adrb2_2RH1.pdb
load structures/DL_03731.sdf
show cartoon, adrb2*
show sticks, DL_*
```

---

## 🎨 PyMOL Commands

### Basic Navigation:
- **Rotate:** Left-click and drag
- **Zoom:** Scroll wheel or right-click drag
- **Pan:** Middle-click drag (or Shift + left-click)

### Useful Commands:
```
# Hide/show objects
disable DL_03731
enable DL_03731

# Color objects
color red, DL_03731
color blue, adrb2_protein

# Change representation
show surface, adrb2_protein
show spheres, DL_03731

# Center view
center DL_03731
zoom DL_03731

# Save image
ray 1200, 900
png my_image.png, dpi=300
```

### Save a Session:
```
save my_session.pse
```

---

## 📸 Creating Publication-Quality Images

```pymol
# Set up nice rendering
bg_color white
set ray_shadows, 1
set ray_trace_mode, 1
set antialias, 2

# Render high-quality image
ray 2400, 1800
png figure.png, dpi=300
```

---

## 🔍 Analyzing Binding Sites

### View Key Residues:
```pymol
select binding_site, resi 113+116+193+194+204+207+296+309+312+313
show sticks, binding_site
color yellow, binding_site
```

### Measure Distances:
```pymol
distance interaction, DL_03731, binding_site, 3.5
```

### Show H-bonds:
```pymol
set h_bond_cutoff_center, 3.5
set h_bond_cutoff_edge, 3.5
show h_bonds, DL_03731
```

---

## 🌐 Alternative Viewers

If you don't have PyMOL, you can use:

### 1. Online Viewers:
- **Mol* Viewer**: https://molstar.org/viewer/
- **3Dmol.js**: https://3dmol.csb.pitt.edu/viewer.html
  - Upload the PDB/SDF files directly

### 2. Free Desktop Software:
- **UCSF Chimera**: https://www.cgl.ucsf.edu/chimera/
- **VMD**: https://www.ks.uiuc.edu/Research/vmd/
- **Avogadro**: https://avogadro.cc/

---

## 💡 Tips

1. **Start simple**: Load just one molecule first
2. **Learn shortcuts**: PyMOL wiki has great tutorials
3. **Save your work**: Use `save session.pse` frequently
4. **Experiment**: Try different colors and representations

---

## 📊 What to Look For

When visualizing your candidates:

1. **Size and Shape**: Do they fit in the binding pocket?
2. **Hydrophobic Regions**: Match with protein hydrophobic patches?
3. **H-bond Donors/Acceptors**: Align with protein residues?
4. **Overall Geometry**: Similar to known ADRB2 ligands?

---

## 🆘 Troubleshooting

**PyMOL won't open:**
```bash
# Try
python3 -m pymol

# Or check installation
which pymol
```

**Files not loading:**
- Make sure you're in the correct directory
- Use absolute paths if needed
- Check file permissions

**Need help?**
- PyMOL Wiki: https://pymolwiki.org/
- PyMOL Forums: https://sourceforge.net/p/pymol/mailman/

---

## 📚 Resources

- **PyMOL Tutorial**: https://pymolwiki.org/index.php/Practical_Pymol_for_Beginners
- **PDB Database**: https://www.rcsb.org/
- **ADRB2 Info**: https://www.rcsb.org/structure/2RH1
