#!/usr/bin/env python3
"""
AutoDock Vina Docking Automation
Handles receptor preparation, docking execution, and results analysis
"""

import os
import subprocess
import pandas as pd
import glob
from pathlib import Path

class VinaDocking:
    """
    Automate AutoDock Vina docking workflow
    """
    
    def __init__(self, project_dir='../'):
        self.project_dir = project_dir
        self.docking_dir = os.path.join(project_dir, 'docking')
        self.results_dir = os.path.join(project_dir, 'results')
        self.protein_dir = os.path.join(project_dir, 'protein')
        
        os.makedirs(self.docking_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
    def prepare_receptor(self, receptor_pdb='beta2_receptor.pdb'):
        """
        Convert receptor PDB to PDBQT format using OpenBabel
        
        Args:
            receptor_pdb: Name of receptor PDB file
            
        Returns:
            Path to prepared receptor PDBQT file
        """
        receptor_path = os.path.join(self.protein_dir, receptor_pdb)
        
        if not os.path.exists(receptor_path):
            raise FileNotFoundError(
                f"Receptor file not found: {receptor_path}\n"
                f"Please place your ADRB2 receptor structure (PDB format) in the protein/ directory"
            )
        
        output_pdbqt = os.path.join(self.docking_dir, 'receptor.pdbqt')
        
        print(f"Preparing receptor: {receptor_pdb}")
        print(f"Input: {receptor_path}")
        print(f"Output: {output_pdbqt}")
        
        # Check if OpenBabel is available
        try:
            subprocess.run(['obabel', '-V'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\nWarning: OpenBabel not found. Install with:")
            print("  sudo apt-get install openbabel")
            print("\nManual preparation required:")
            print(f"  obabel {receptor_path} -O {output_pdbqt} -p 7.4 -xh")
            return None
        
        # Convert PDB to PDBQT
        cmd = [
            'obabel',
            receptor_path,
            '-O', output_pdbqt,
            '-p', '7.4',  # pH for protonation
            '-xh'  # Add hydrogens
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"\nReceptor prepared successfully: {output_pdbqt}")
            return output_pdbqt
        except subprocess.CalledProcessError as e:
            print(f"Error preparing receptor: {e}")
            print(f"stderr: {e.stderr}")
            return None
    
    def create_config_file(self, center_x=0.0, center_y=0.0, center_z=0.0,
                          size_x=25.0, size_y=25.0, size_z=25.0,
                          exhaustiveness=8, num_modes=9):
        """
        Create AutoDock Vina configuration file
        
        Args:
            center_x, center_y, center_z: Grid box center coordinates
            size_x, size_y, size_z: Grid box dimensions (Angstroms)
            exhaustiveness: Search exhaustiveness (higher = more thorough)
            num_modes: Number of binding modes to generate
            
        Returns:
            Path to config file
        """
        config_path = os.path.join(self.docking_dir, 'vina_config.txt')
        
        receptor_path = os.path.join(self.docking_dir, 'receptor.pdbqt')
        
        config_content = f"""# AutoDock Vina Configuration File
# ADRB2 Ligand Docking

receptor = {receptor_path}

# Grid box center (binding site coordinates)
center_x = {center_x}
center_y = {center_y}
center_z = {center_z}

# Grid box size (search space in Angstroms)
size_x = {size_x}
size_y = {size_y}
size_z = {size_z}

# Search parameters
exhaustiveness = {exhaustiveness}
num_modes = {num_modes}

# Energy range for saved conformations (kcal/mol)
energy_range = 3
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"\nConfiguration file created: {config_path}")
        print(f"Grid box center: ({center_x}, {center_y}, {center_z})")
        print(f"Grid box size: {size_x} x {size_y} x {size_z} Å")
        
        return config_path
    
    def run_docking(self, ligand_dir='ligands', config_file='vina_config.txt'):
        """
        Run AutoDock Vina for all ligands in specified directory
        
        Args:
            ligand_dir: Directory containing ligand PDBQT files
            config_file: Vina configuration file
            
        Returns:
            List of docking results
        """
        ligands_path = os.path.join(self.docking_dir, ligand_dir)
        config_path = os.path.join(self.docking_dir, config_file)
        
        if not os.path.exists(ligands_path):
            raise FileNotFoundError(f"Ligands directory not found: {ligands_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Find all ligand files
        ligand_files = glob.glob(os.path.join(ligands_path, '*.pdbqt'))
        
        if not ligand_files:
            raise FileNotFoundError(f"No PDBQT ligand files found in {ligands_path}")
        
        print(f"\nFound {len(ligand_files)} ligands to dock")
        
        # Check if Vina is available
        try:
            subprocess.run(['vina', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\nWarning: AutoDock Vina not found. Install instructions:")
            print("  Linux: sudo apt-get install autodock-vina")
            print("  Mac: brew install autodock-vina")
            print("  Or download from: https://vina.scripps.edu/")
            print("\nManual docking command:")
            print(f"  vina --config {config_path} --ligand <LIGAND>.pdbqt --out <RESULT>.pdbqt")
            return []
        
        results = []
        output_dir = os.path.join(self.docking_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        for i, ligand_file in enumerate(ligand_files, 1):
            ligand_name = Path(ligand_file).stem
            output_file = os.path.join(output_dir, f'{ligand_name}_docked.pdbqt')
            log_file = os.path.join(output_dir, f'{ligand_name}_log.txt')
            
            print(f"\n[{i}/{len(ligand_files)}] Docking {ligand_name}...")
            
            cmd = [
                'vina',
                '--config', config_path,
                '--ligand', ligand_file,
                '--out', output_file
            ]
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    timeout=300  # 5 minute timeout per ligand
                )
                
                # Save log
                with open(log_file, 'w') as f:
                    f.write(result.stdout)
                
                # Parse binding affinity
                binding_affinity = self.parse_vina_output(result.stdout)
                
                results.append({
                    'ligand': ligand_name,
                    'binding_affinity': binding_affinity,
                    'output_file': output_file,
                    'log_file': log_file
                })
                
                print(f"  Binding affinity: {binding_affinity} kcal/mol")
                
            except subprocess.TimeoutExpired:
                print(f"  Timeout - skipping {ligand_name}")
                continue
            except subprocess.CalledProcessError as e:
                print(f"  Error docking {ligand_name}: {e}")
                print(f"  stderr: {e.stderr}")
                continue
        
        print(f"\n=== Docking Complete ===")
        print(f"Successfully docked: {len(results)}/{len(ligand_files)} ligands")
        
        return results
    
    def parse_vina_output(self, output_text):
        """
        Extract best binding affinity from Vina output
        
        Args:
            output_text: Vina stdout text
            
        Returns:
            Best binding affinity (kcal/mol)
        """
        for line in output_text.split('\n'):
            if line.strip().startswith('1 '):
                # First mode has best affinity
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
        return None
    
    def analyze_results(self, docking_results, ai_predictions_file='ai_predictions.csv'):
        """
        Combine AI predictions with docking results
        
        Args:
            docking_results: List of docking result dictionaries
            ai_predictions_file: CSV file with AI predictions
            
        Returns:
            DataFrame with combined results
        """
        if not docking_results:
            print("No docking results to analyze")
            return None
        
        # Load AI predictions
        predictions_path = os.path.join(self.results_dir, ai_predictions_file)
        ai_df = pd.read_csv(predictions_path)
        
        # Create docking dataframe
        dock_df = pd.DataFrame(docking_results)
        
        # Extract molecule IDs from ligand names
        dock_df['molecule_id'] = dock_df['ligand'].str.replace('ligand_', '')
        
        # Merge with AI predictions
        combined = pd.merge(
            ai_df,
            dock_df[['molecule_id', 'binding_affinity', 'output_file']],
            on='molecule_id',
            how='inner'
        )
        
        # Calculate composite score (normalized)
        # Higher predicted_pIC50 is better, lower (more negative) binding affinity is better
        combined['ai_score_norm'] = (combined['predicted_pIC50'] - combined['predicted_pIC50'].min()) / \
                                     (combined['predicted_pIC50'].max() - combined['predicted_pIC50'].min())
        
        combined['dock_score_norm'] = (combined['binding_affinity'].max() - combined['binding_affinity']) / \
                                       (combined['binding_affinity'].max() - combined['binding_affinity'].min())
        
        # Composite score (equal weighting)
        combined['composite_score'] = (combined['ai_score_norm'] + combined['dock_score_norm']) / 2
        
        # Sort by composite score
        combined = combined.sort_values('composite_score', ascending=False)
        
        # Save results
        output_path = os.path.join(self.results_dir, 'final_hits.csv')
        combined.to_csv(output_path, index=False)
        
        print(f"\n=== Final Analysis ===")
        print(f"Combined {len(combined)} molecules with both AI and docking scores")
        print(f"Results saved to: {output_path}")
        print("\nTop 5 candidates:")
        print(combined[['molecule_id', 'predicted_pIC50', 'binding_affinity', 
                       'composite_score']].head().to_string(index=False))
        
        return combined


def main():
    """
    Main docking workflow
    """
    print("=" * 60)
    print("AutoDock Vina Docking Workflow")
    print("=" * 60)
    
    vina = VinaDocking(project_dir='/home/claude/adrb2_discovery')
    
    # Step 1: Prepare receptor
    print("\n--- STEP 1: RECEPTOR PREPARATION ---\n")
    receptor = vina.prepare_receptor()
    
    if receptor is None:
        print("\nReceptor preparation required before proceeding")
        return
    
    # Step 2: Create config file
    print("\n--- STEP 2: CONFIGURATION ---\n")
    print("Note: Update grid box coordinates based on your binding site!")
    print("Default configuration uses center (0, 0, 0) - ADJUST AS NEEDED")
    
    config = vina.create_config_file(
        center_x=0.0,  # UPDATE THESE
        center_y=0.0,  # VALUES BASED ON
        center_z=0.0,  # YOUR BINDING SITE
        size_x=25.0,
        size_y=25.0,
        size_z=25.0,
        exhaustiveness=8
    )
    
    # Step 3: Run docking
    print("\n--- STEP 3: MOLECULAR DOCKING ---\n")
    results = vina.run_docking()
    
    if results:
        # Step 4: Analyze results
        print("\n--- STEP 4: RESULTS ANALYSIS ---\n")
        final_results = vina.analyze_results(results)
        
        print("\n" + "=" * 60)
        print("Docking Complete!")
        print("=" * 60)
        print("\nReview final_hits.csv for top candidates")
        print("Strong hits have:")
        print("  - High predicted_pIC50 (>6.0)")
        print("  - Low binding_affinity (<-7.0 kcal/mol)")
        print("  - High composite_score (>0.7)")


if __name__ == "__main__":
    main()
