import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from meeko import PDBQTMolecule

def prepare_ligands(input_csv, output_dir, smiles_col):
    print(f"Loading molecules from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found.")
        sys.exit(1)

    df = pd.read_csv(input_csv)

    if smiles_col not in df.columns:
        print(f"Error: Column '{smiles_col}' not found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for idx, row in df.iterrows():
        smi = row[smiles_col]
        mol_name = row.get('molecule_name', f"mol_{idx}")
        # Sanitize name for filename
        safe_name = "".join([c if c.isalnum() else "_" for c in mol_name])
        output_file = os.path.join(output_dir, f"{safe_name}.pdbqt")

        print(f"Processing {mol_name}...")

        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"Warning: Invalid SMILES for {mol_name}. Skipping.")
                continue

            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D conformer
            res = AllChem.EmbedMolecule(mol, randomSeed=42)
            if res != 0:
                print(f"Warning: Failed to embed molecule {mol_name}. Skipping.")
                continue

            AllChem.MMFFOptimizeMolecule(mol)

            # Prepare for docking using Meeko
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            pdbqt_string = preparator.write_pdbqt_string()

            with open(output_file, 'w') as f:
                f.write(pdbqt_string)

            count += 1

        except Exception as e:
            print(f"Error processing {mol_name}: {e}")

    print(f"Successfully prepared {count} ligands in {output_dir}")

if __name__ == "__main__":
    input_csv = "ai_assisted_drug_design/data/new_molecules/sample_test.csv"
    output_dir = "ai_assisted_drug_design/ligands/pdbqt"
    smiles_col = "canonical_smiles"

    prepare_ligands(input_csv, output_dir, smiles_col)
