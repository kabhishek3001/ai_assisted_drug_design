import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import argparse
import os
import sys

# Configuration
MODEL_PATH = "ai_assisted_drug_design/ai_model/model.pkl"

def generate_fingerprints(smiles_list):
    """Generates Morgan fingerprints for a list of SMILES strings."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]

    if not valid_indices:
        return np.array([]), []

    # Filter out invalid molecules
    valid_mols = [mols[i] for i in valid_indices]

    # Generate fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in valid_mols]
    X = np.array(fps)

    return X, valid_indices

def predict(input_file, output_file, smiles_col):
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please train the model first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    print(f"Loading input data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)

    df = pd.read_csv(input_file)

    if smiles_col not in df.columns:
        print(f"Error: Column '{smiles_col}' not found in input file.")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    smiles = df[smiles_col].tolist()
    print(f"Processing {len(smiles)} molecules...")

    X, valid_indices = generate_fingerprints(smiles)

    if len(valid_indices) == 0:
        print("Error: No valid molecules found in input.")
        sys.exit(1)

    print(f"Generated fingerprints for {len(valid_indices)} valid molecules.")

    # Predict
    predictions = model.predict(X)

    # Create result dataframe
    # We want to keep original data, but only for valid molecules
    result_df = df.iloc[valid_indices].copy()
    result_df['predicted_pchembl_value'] = predictions
    result_df['predicted_affinity_type'] = result_df['predicted_pchembl_value'].apply(
        lambda x: 'High' if x >= 7 else ('Medium' if x >= 5 else 'Low')
    )

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Print preview
    print("\nPrediction Preview:")
    print(result_df[[smiles_col, 'predicted_pchembl_value', 'predicted_affinity_type']].head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict binding affinity for new molecules.")
    parser.add_argument("--input", required=True, help="Path to input CSV file containing SMILES.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--smiles_col", default="canonical_smiles", help="Name of the column containing SMILES strings.")

    args = parser.parse_args()

    predict(args.input, args.output, args.smiles_col)
