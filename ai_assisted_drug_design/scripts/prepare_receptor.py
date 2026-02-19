import os
import sys
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import warnings
import subprocess

# Suppress OpenMM warnings
warnings.simplefilter('ignore')

def prepare_receptor(input_pdb, output_pdbqt):
    print(f"Preparing receptor from {input_pdb}...")

    # 1. Load PDB using PDBFixer
    fixer = PDBFixer(filename=input_pdb)

    # 2. Find missing residues and atoms (standard cleaning)
    print("Finding missing residues...")
    fixer.findMissingResidues()
    print("Finding missing atoms...")
    fixer.findMissingAtoms()

    # 3. Add missing atoms
    print("Adding missing atoms...")
    fixer.addMissingAtoms()

    # 4. Remove water and heterogens (keep only protein)
    # PDBFixer keeps heterogens by default, we need to filter them if we want pure protein
    print("Removing water and heterogens (keeping protein only)...")
    fixer.removeHeterogens(keepWater=False)

    # 5. Add hydrogens (pH 7.4)
    print("Adding hydrogens (pH 7.4)...")
    fixer.addMissingHydrogens(7.4)

    # 6. Save intermediate PDB (cleaned and protonated)
    intermediate_pdb = input_pdb.replace(".pdb", "_fixed.pdb")
    print(f"Saving intermediate fixed PDB to {intermediate_pdb}...")
    with open(intermediate_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    # 7. Convert to PDBQT using OpenBabel
    # We use -xr to keep rigid residues (standard for receptor) and -xn to preserve atom names if possible.
    # OpenBabel adds Gasteiger charges automatically for pdbqt output.
    print(f"Converting to PDBQT using OpenBabel: {output_pdbqt}")
    cmd = [
        "obabel",
        intermediate_pdb,
        "-O", output_pdbqt,
        "-xr", # Output as rigid molecule
        "--partialcharge", "gasteiger"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Receptor successfully prepared at {output_pdbqt}")
    except subprocess.CalledProcessError as e:
        print(f"Error running OpenBabel: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'obabel' command not found. Please install OpenBabel.")
        sys.exit(1)

if __name__ == "__main__":
    input_path = "ai_assisted_drug_design/protein/beta2_receptor.pdb"
    output_path = "ai_assisted_drug_design/protein/prepared/beta2_receptor.pdbqt"

    if not os.path.exists(input_path):
        print(f"Error: Input PDB {input_path} not found.")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prepare_receptor(input_path, output_path)
