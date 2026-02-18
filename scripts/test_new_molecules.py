#!/usr/bin/env python3
"""
Test New Molecules Pipeline
Automated workflow to test a set of new molecules against ADRB2 using AI predictions and Docking.
"""

import os
import sys
import pandas as pd
import requests
import subprocess
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

# Add script directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_pipeline import ADRB2DrugDiscovery
from run_docking import VinaDocking
from visualize_results import ResultsVisualizer

class NewMoleculeTester:
    def __init__(self, project_dir='../'):
        self.project_dir = os.path.abspath(project_dir)
        self.data_dir = os.path.join(self.project_dir, 'data')
        self.models_dir = os.path.join(self.project_dir, 'models')
        self.results_dir = os.path.join(self.project_dir, 'results')
        self.protein_dir = os.path.join(self.project_dir, 'protein')

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.protein_dir, exist_ok=True)

    def check_and_train_model(self):
        """Check if AI model exists, if not train it."""
        model_path = os.path.join(self.models_dir, 'adrb2_rf_model.pkl')
        if not os.path.exists(model_path):
            print(f"AI Model not found at {model_path}. Training now...")
            pipeline = ADRB2DrugDiscovery(project_dir=self.project_dir)

            # Check if training data exists
            chembl_file = os.path.join(self.data_dir, 'chembl_adrb2_data.csv')
            if not os.path.exists(chembl_file):
                print("ChEMBL training data not found. Fetching data...")
                # Run fetch_data.py
                fetch_script = os.path.join(self.project_dir, 'scripts', 'fetch_data.py')
                subprocess.run(['python3', fetch_script], check=True)

            pipeline.load_and_prepare_data()
            pipeline.train_model()
        else:
            print("AI Model found.")

    def check_and_download_receptor(self):
        """Check if receptor PDB exists, if not download 2RH1."""
        receptor_path = os.path.join(self.protein_dir, 'beta2_receptor.pdb')
        if not os.path.exists(receptor_path):
            print(f"Receptor file not found. Downloading PDB: 2RH1 to {receptor_path}...")
            url = "https://files.rcsb.org/download/2RH1.pdb"
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(receptor_path, 'wb') as f:
                    f.write(response.content)
                print("Receptor downloaded successfully.")

                # Optional: Clean up PDB (remove waters, keep chain A) - Basic cleanup
                # For now we use the file as is or rely on VinaDocking.prepare_receptor to handle conversion
            except Exception as e:
                print(f"Failed to download receptor: {e}")
                sys.exit(1)
        else:
            print("Receptor file found.")

    def run_prediction_and_docking(self, input_file='new_molecules.csv'):
        """Run the full pipeline on new molecules."""
        print(f"\n--- Processing {input_file} ---")

        # 1. AI Prediction
        pipeline = ADRB2DrugDiscovery(project_dir=self.project_dir)

        # Load model explicitly if needed, but virtual_screening handles it
        try:
            # We use virtual_screening to generate predictions
            # output will be saved to results/ai_predictions.csv
            top_candidates = pipeline.virtual_screening(
                candidates_file=input_file,
                top_n=100, # Process all or top N
                output_file='ai_predictions.csv'
            )
        except Exception as e:
            print(f"Error during AI prediction: {e}")
            return

        # 2. Docking
        print("\n--- Starting Docking Workflow ---")
        vina = VinaDocking(project_dir=self.project_dir)

        # Prepare receptor
        receptor_pdbqt = vina.prepare_receptor()
        if not receptor_pdbqt:
            print("Failed to prepare receptor. Aborting docking.")
            return

        # Prepare ligands from AI predictions
        # We need to prepare ALL or Top N molecules from the new list
        # prepare_for_docking reads from 'ai_predictions.csv' in results_dir
        # Let's dock all of them since the list is small
        prepared_ligands = pipeline.prepare_for_docking(
            predictions_file='ai_predictions.csv',
            top_n=100,  # Dock everything in the new file
            output_dir='ligands'
        )

        if not prepared_ligands:
            print("No ligands prepared for docking.")
            return

        # Run Docking
        # config file should already be set up in docking/vina_config.txt
        docking_results = vina.run_docking()

        # Analyze Results
        final_hits = vina.analyze_results(docking_results)

        # 3. Visualization and Reporting
        print("\n--- Generating Visualizations ---")
        viz = ResultsVisualizer(project_dir=self.project_dir)
        viz.plot_prediction_distribution()
        if docking_results:
            viz.plot_docking_results()
            report = viz.generate_report()
            print("\n" + report)

def main():
    tester = NewMoleculeTester(project_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Ensure dependencies
    tester.check_and_train_model()
    tester.check_and_download_receptor()

    # Run test
    tester.run_prediction_and_docking(input_file='new_molecules.csv')

if __name__ == "__main__":
    main()
