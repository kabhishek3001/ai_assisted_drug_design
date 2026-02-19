import os
import subprocess
import glob
import pandas as pd
import re

def run_docking(receptor_pdbqt, ligand_dir, config_file, output_dir):
    print(f"Running docking using receptor: {receptor_pdbqt}")

    ligand_files = glob.glob(os.path.join(ligand_dir, "*.pdbqt"))
    if not ligand_files:
        print(f"No ligand PDBQT files found in {ligand_dir}")
        return

    results = []

    os.makedirs(output_dir, exist_ok=True)

    for lig_file in ligand_files:
        lig_name = os.path.basename(lig_file).replace(".pdbqt", "")
        output_pdbqt = os.path.join(output_dir, f"{lig_name}_docked.pdbqt")
        log_file = os.path.join(output_dir, f"{lig_name}_docking.log")

        print(f"Docking {lig_name}...")

        cmd = [
            "vina",
            "--receptor", receptor_pdbqt,
            "--ligand", lig_file,
            "--config", config_file,
            "--out", output_pdbqt
        ]

        try:
            # Run Vina and capture output
            with open(log_file, "w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

            # Parse the best binding affinity from the log file
            # Output format example:
            #    1         -9.5      0.000      0.000
            affinity = None
            with open(log_file, "r") as f:
                for line in f:
                    match = re.search(r'^\s*1\s+(-?\d+\.\d+)', line)
                    if match:
                        affinity = float(match.group(1))
                        break

            if affinity is not None:
                print(f"  Score: {affinity} kcal/mol")
                results.append({"molecule_name": lig_name, "docking_score": affinity})
            else:
                print(f"  Warning: Could not parse affinity for {lig_name}")

        except subprocess.CalledProcessError as e:
            print(f"  Error docking {lig_name}: {e}")

    # Save summary
    if results:
        df = pd.DataFrame(results)
        summary_file = os.path.join(output_dir, "docking_summary.csv")
        df.sort_values("docking_score", inplace=True)
        df.to_csv(summary_file, index=False)
        print(f"\nDocking completed. Summary saved to {summary_file}")
        print(df)
    else:
        print("No results to save.")

if __name__ == "__main__":
    receptor = "ai_assisted_drug_design/protein/prepared/beta2_receptor.pdbqt"
    ligand_dir = "ai_assisted_drug_design/ligands/pdbqt"
    config = "ai_assisted_drug_design/docking/vina_config.txt"
    output_dir = "ai_assisted_drug_design/docking/results"

    if not os.path.exists(receptor):
        print(f"Error: Receptor file {receptor} not found.")
    elif not os.path.exists(config):
        print(f"Error: Config file {config} not found.")
    else:
        run_docking(receptor, ligand_dir, config, output_dir)
