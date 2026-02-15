#!/usr/bin/env python3
"""
PyMOL Visualization Script for ADRB2 Drug Discovery

This script:
1. Downloads ADRB2 protein structure from PDB
2. Generates 3D structures for top candidates
3. Creates PyMOL visualization scripts
4. Enables interactive exploration of protein-ligand binding
"""

import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import requests

class ADRB2Visualizer:
    """Create visualizations for ADRB2 and drug candidates"""
    
    def __init__(self, project_dir='../'):
        self.project_dir = project_dir
        self.results_dir = os.path.join(project_dir, 'results')
        self.viz_dir = os.path.join(project_dir, 'visualizations')
        self.pdb_dir = os.path.join(self.viz_dir, 'structures')
        
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.pdb_dir, exist_ok=True)
    
    def download_adrb2_structure(self):
        """Download ADRB2 structure from PDB"""
        print("\n" + "="*70)
        print("Downloading ADRB2 Protein Structure")
        print("="*70)
        
        # ADRB2 PDB IDs (crystal structures)
        pdb_ids = {
            '2RH1': 'ADRB2 with carazolol (inverse agonist)',
            '3SN6': 'ADRB2 with BI-167107 (agonist) + G-protein',
            '3P0G': 'ADRB2 with ICI-118551 (antagonist)',
            '4LDO': 'ADRB2 with salmeterol (agonist)'
        }
        
        print("\nAvailable ADRB2 structures:")
        for pdb_id, description in pdb_ids.items():
            print(f"  {pdb_id}: {description}")
        
        # Download the most common one (2RH1)
        pdb_id = '2RH1'
        print(f"\nDownloading {pdb_id}...")
        
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_file = os.path.join(self.pdb_dir, f'adrb2_{pdb_id}.pdb')
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_file, 'w') as f:
                f.write(response.text)
            
            print(f"✓ Downloaded: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"✗ Error downloading PDB: {e}")
            print("\nAlternative: Download manually from:")
            print(f"  https://www.rcsb.org/structure/{pdb_id}")
            return None
    
    def generate_3d_structures(self, top_n=10):
        """Generate 3D structures for top candidates"""
        print("\n" + "="*70)
        print("Generating 3D Structures for Top Candidates")
        print("="*70)
        
        # Load predictions
        predictions_file = os.path.join(self.results_dir, 'ai_predictions.csv')
        df = pd.read_csv(predictions_file)
        
        # Take top N
        top_molecules = df.head(top_n)
        
        print(f"\nGenerating 3D structures for top {top_n} molecules...")
        
        generated_files = []
        
        for idx, row in top_molecules.iterrows():
            mol_id = row['molecule_id']
            smiles = row['smiles']
            pic50 = row['predicted_pIC50']
            
            print(f"\n  {idx+1}. {mol_id} (pIC50={pic50:.2f})")
            
            try:
                # Convert SMILES to 3D
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"     ✗ Invalid SMILES")
                    continue
                
                # Add hydrogens
                mol = Chem.AddHs(mol)
                
                # Generate 3D coordinates
                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result != 0:
                    print(f"     ✗ Could not generate 3D coordinates")
                    continue
                
                # Optimize geometry
                AllChem.MMFFOptimizeMolecule(mol)
                
                # Save as PDB
                output_file = os.path.join(self.pdb_dir, f'{mol_id}.pdb')
                Chem.MolToPDBFile(mol, output_file)
                
                # Also save as SDF (better format for small molecules)
                sdf_file = os.path.join(self.pdb_dir, f'{mol_id}.sdf')
                writer = Chem.SDWriter(sdf_file)
                writer.write(mol)
                writer.close()
                
                generated_files.append({
                    'mol_id': mol_id,
                    'pdb_file': output_file,
                    'sdf_file': sdf_file,
                    'pic50': pic50
                })
                
                print(f"     ✓ Generated: {mol_id}.pdb and {mol_id}.sdf")
                
            except Exception as e:
                print(f"     ✗ Error: {e}")
                continue
        
        print(f"\n✓ Generated {len(generated_files)} 3D structures")
        return generated_files
    
    def create_pymol_script(self, protein_file, ligand_files):
        """Create PyMOL visualization script"""
        print("\n" + "="*70)
        print("Creating PyMOL Visualization Script")
        print("="*70)
        
        script_path = os.path.join(self.viz_dir, 'visualize_adrb2.pml')
        
        script = f"""# PyMOL Visualization Script for ADRB2 Drug Discovery
# Generated automatically

# Clean slate
reinitialize

# Load ADRB2 protein structure
load {protein_file}, adrb2_protein

# Style the protein
hide everything, adrb2_protein
show cartoon, adrb2_protein
color marine, adrb2_protein
set cartoon_fancy_helices, 1

# Show binding site residues as sticks
select binding_site, resi 113+116+117+120+193+194+197+200+201+204+207+208+296+299+300+305+309+312+313
show sticks, binding_site
color yellow, binding_site
set stick_radius, 0.15, binding_site

# Load top candidate molecules
"""
        
        colors = ['red', 'green', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'lime', 'salmon', 'lightblue']
        
        for i, lig in enumerate(ligand_files[:10]):
            mol_id = lig['mol_id']
            sdf_file = lig['sdf_file']
            pic50 = lig['pic50']
            color = colors[i % len(colors)]
            
            script += f"""
# Candidate {i+1}: {mol_id} (pIC50={pic50:.2f})
load {sdf_file}, {mol_id}
hide everything, {mol_id}
show sticks, {mol_id}
color {color}, {mol_id}
set stick_radius, 0.2, {mol_id}
"""
        
        script += """
# Center view on binding site
center binding_site
zoom binding_site, 8

# Set up nice rendering
set ray_shadows, 0
set antialias, 2
set line_smooth, 1
set depth_cue, 1
bg_color white

# Create different views
set_view (\\
     0.999,    0.000,    0.000,\\
     0.000,    0.999,    0.000,\\
     0.000,    0.000,    0.999,\\
     0.000,    0.000, -150.000,\\
     0.000,    0.000,    0.000,\\
    50.000,  250.000,    0.000 )

# Instructions
print ""
print "="*70
print "PyMOL Visualization Loaded!"
print "="*70
print ""
print "Commands:"
print "  - Click and drag to rotate"
print "  - Shift + drag to zoom"
print "  - Use mouse wheel to zoom in/out"
print ""
print "Objects:"
print "  - adrb2_protein: The ADRB2 receptor"
print "  - binding_site: Key binding site residues (yellow)"
"""
        
        for i, lig in enumerate(ligand_files[:10]):
            script += f"""print "  - {lig['mol_id']}: Candidate {i+1} (pIC50={lig['pic50']:.2f}, {colors[i % len(colors)]})"
"""
        
        script += """
print ""
print "Toggle visibility:"
"""
        
        for lig in ligand_files[:10]:
            script += f"""print "  disable {lig['mol_id']}  # Hide candidate"
"""
        
        script += """
print ""
print "Export images:"
print "  ray 1200, 900"
print "  png adrb2_view.png, dpi=300"
print ""
print "="*70
"""
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        print(f"✓ Created PyMOL script: {script_path}")
        return script_path
    
    def create_simple_visualization_script(self, ligand_files):
        """Create simple script to view just the molecules"""
        print("\nCreating simple molecule viewer script...")
        
        script_path = os.path.join(self.viz_dir, 'view_molecules.pml')
        
        script = """# Simple Molecule Viewer
# View top drug candidates

reinitialize

"""
        colors = ['red', 'green', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'lime']
        
        for i, lig in enumerate(ligand_files[:10]):
            mol_id = lig['mol_id']
            sdf_file = lig['sdf_file']
            pic50 = lig['pic50']
            color = colors[i % len(colors)]
            
            script += f"""
# {mol_id} (pIC50={pic50:.2f})
load {sdf_file}, {mol_id}
show sticks, {mol_id}
color {color}, {mol_id}
"""
        
        script += """
# Nice rendering
set stick_radius, 0.15
bg_color white
set antialias, 2
zoom

print "Loaded top drug candidates!"
print "Use 'disable molecule_name' to hide a molecule"
print "Use 'enable molecule_name' to show it again"
"""
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        print(f"✓ Created simple viewer: {script_path}")
        return script_path
    
    def create_readme(self, protein_file, pymol_script, simple_script):
        """Create README with instructions"""
        readme_path = os.path.join(self.viz_dir, 'README.md')
        
        readme = f"""# ADRB2 Visualization Guide

## 📁 Files Generated

- **Protein Structure:**
  - `{os.path.basename(protein_file)}` - ADRB2 receptor structure from PDB

- **Top Candidate Structures:**
  - `structures/*.pdb` - Individual molecule files (PDB format)
  - `structures/*.sdf` - Individual molecule files (SDF format)

- **PyMOL Scripts:**
  - `{os.path.basename(pymol_script)}` - Full protein + ligands visualization
  - `{os.path.basename(simple_script)}` - Simple molecule viewer

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
pymol {os.path.basename(pymol_script)}
```

#### Or Load Just Molecules:
```bash
pymol {os.path.basename(simple_script)}
```

### Method 2: Using PyMOL GUI

1. Open PyMOL
2. File → Run Script
3. Select `{pymol_script}`

### Method 3: Manual Loading

In PyMOL command line:
```
load {protein_file}
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
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme)
        
        print(f"✓ Created README: {readme_path}")
        return readme_path


def main():
    """Main execution"""
    print("="*70)
    print("ADRB2 Visualization Setup")
    print("="*70)
    
    # Get project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    viz = ADRB2Visualizer(project_dir=project_dir)
    
    # Step 1: Download protein structure
    protein_file = viz.download_adrb2_structure()
    
    # Step 2: Generate 3D structures for candidates
    ligand_files = viz.generate_3d_structures(top_n=10)
    
    if not ligand_files:
        print("\n✗ No ligand structures generated")
        return
    
    # Step 3: Create PyMOL scripts
    if protein_file:
        pymol_script = viz.create_pymol_script(protein_file, ligand_files)
    else:
        pymol_script = None
    
    simple_script = viz.create_simple_visualization_script(ligand_files)
    
    # Step 4: Create README
    viz.create_readme(
        protein_file if protein_file else "adrb2_2RH1.pdb",
        pymol_script if pymol_script else "visualize_adrb2.pml",
        simple_script
    )
    
    # Final summary
    print("\n" + "="*70)
    print("✅ VISUALIZATION SETUP COMPLETE!")
    print("="*70)
    print(f"\nFiles created in: {viz.viz_dir}/")
    print(f"  - Protein structure: structures/adrb2_2RH1.pdb")
    print(f"  - {len(ligand_files)} candidate structures: structures/DL_*.pdb")
    print(f"  - PyMOL scripts: *.pml")
    print(f"  - Instructions: README.md")
    
    print("\n" + "="*70)
    print("🚀 NEXT STEPS")
    print("="*70)
    print("\n1. Install PyMOL:")
    print("   sudo apt install pymol")
    print("\n2. Visualize your molecules:")
    print(f"   cd {viz.viz_dir}")
    print("   pymol visualize_adrb2.pml")
    print("\n3. Or view just molecules:")
    print("   pymol view_molecules.pml")
    print("\n4. Read the README for full instructions:")
    print(f"   cat {os.path.join(viz.viz_dir, 'README.md')}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()