import os
import sys
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_model.predict import generate_fingerprints
import joblib
from scripts.prepare_ligands import prepare_ligands
from scripts.run_docking import run_docking

def virtual_screening(input_csv, output_dir, model_path, receptor_path, docking_config, top_n=5, smiles_col='canonical_smiles'):
    print("=== AI-Assisted Virtual Screening Pipeline ===")
    print(f"Input Data: {input_csv}")

    # 1. Load Data
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found.")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} molecules.")

    if smiles_col not in df.columns:
        print(f"Error: Column '{smiles_col}' not found.")
        sys.exit(1)

    # 2. AI Prediction
    print("\n--- Step 1: AI Prediction ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)

    model = joblib.load(model_path)
    X, valid_indices = generate_fingerprints(df[smiles_col].tolist())

    if len(valid_indices) == 0:
        print("Error: No valid molecules for prediction.")
        sys.exit(1)

    predictions = model.predict(X)

    # Add predictions to dataframe (handling invalid molecules)
    df_valid = df.iloc[valid_indices].copy()
    df_valid['ai_score'] = predictions

    # Sort by AI Score (Descending for pChEMBL - higher is better affinity)
    df_sorted = df_valid.sort_values('ai_score', ascending=False)

    print(f"Top 5 AI Predictions:")
    print(df_sorted[[smiles_col, 'ai_score']].head())

    # 3. Select Top Candidates
    top_candidates = df_sorted.head(top_n).copy()
    print(f"\nSelected top {len(top_candidates)} candidates for docking.")

    # Save intermediate candidates
    os.makedirs(output_dir, exist_ok=True)
    candidates_file = os.path.join(output_dir, "top_candidates.csv")
    top_candidates.to_csv(candidates_file, index=False)

    # 4. Ligand Preparation
    print("\n--- Step 2: Ligand Preparation ---")
    ligand_prep_dir = os.path.join(output_dir, "ligands_prepared")
    # Clean previous preparation if needed, or just overwrite
    prepare_ligands(candidates_file, ligand_prep_dir, smiles_col)

    # 5. Molecular Docking
    print("\n--- Step 3: Molecular Docking ---")
    docking_results_dir = os.path.join(output_dir, "docking_results")
    run_docking(receptor_path, ligand_prep_dir, docking_config, docking_results_dir)

    # 6. Merge Results
    print("\n--- Step 4: Analysis & Reporting ---")
    docking_summary_file = os.path.join(docking_results_dir, "docking_summary.csv")

    if os.path.exists(docking_summary_file):
        docking_df = pd.read_csv(docking_summary_file)

        # Merge on molecule name (which prepare_ligands generated)
        # We need to reconstruct names in top_candidates to match prepare_ligands logic
        # prepare_ligands uses 'molecule_name' column or 'mol_{idx}'
        # Let's add a name column to top_candidates if missing
        if 'molecule_name' not in top_candidates.columns:
             top_candidates['molecule_name'] = [f"mol_{i}" for i in top_candidates.index]

        # Sanitize names as per prepare_ligands
        top_candidates['sanitized_name'] = top_candidates['molecule_name'].apply(
            lambda x: "".join([c if c.isalnum() else "_" for c in str(x)])
        )

        # Merge
        final_df = pd.merge(top_candidates, docking_df, left_on='sanitized_name', right_on='molecule_name', how='inner')

        # Rename columns for clarity
        final_df = final_df.rename(columns={'docking_score': 'docking_score_kcal_mol', 'ai_score': 'ai_pchembl_value'})

        # Select key columns
        cols = ['molecule_name_x', smiles_col, 'ai_pchembl_value', 'docking_score_kcal_mol']
        final_report = final_df[cols].sort_values('docking_score_kcal_mol') # Sort by best docking score (lowest is best)

        report_path = os.path.join(output_dir, "final_screening_report.csv")
        final_report.to_csv(report_path, index=False)
        print(f"Final report saved to {report_path}")
        print("\nTop Docked Candidates:")
        print(final_report)

        # Correlation Analysis
        if len(final_report) > 2:
            try:
                # AI Score is pChEMBL (higher is better). Docking score is kcal/mol (lower is better).
                # Expect NEGATIVE correlation if they agree.
                corr, _ = pearsonr(final_report['ai_pchembl_value'], final_report['docking_score_kcal_mol'])
                print(f"\nCorrelation between AI and Docking Scores: {corr:.4f}")

                # Plot
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=final_report, x='ai_pchembl_value', y='docking_score_kcal_mol')
                plt.title(f"AI Prediction vs. Docking Score (n={len(final_report)})")
                plt.xlabel("AI Predicted pChEMBL")
                plt.ylabel("Docking Score (kcal/mol)")
                plt.grid(True)
                plot_path = os.path.join(output_dir, "../plots/correlation.png")
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path)
                print(f"Correlation plot saved to {plot_path}")
            except Exception as e:
                print(f"Could not generate plot: {e}")

    else:
        print("Warning: No docking results found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI-Assisted Virtual Screening Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV with molecules")
    parser.add_argument("--output_dir", default="ai_assisted_drug_design/results/screening_run", help="Output directory")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top AI candidates to dock")

    args = parser.parse_args()

    # Paths (relative to project root assumption)
    MODEL_PATH = "ai_assisted_drug_design/ai_model/model.pkl"
    RECEPTOR_PATH = "ai_assisted_drug_design/protein/prepared/beta2_receptor.pdbqt"
    CONFIG_PATH = "ai_assisted_drug_design/docking/vina_config.txt"

    virtual_screening(args.input, args.output_dir, MODEL_PATH, RECEPTOR_PATH, CONFIG_PATH, args.top_n)
