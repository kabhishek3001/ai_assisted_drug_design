import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Configuration
DATA_PATH = "ai_assisted_drug_design/data/raw/chembl_adrb2.csv"
MODEL_PATH = "ai_assisted_drug_design/ai_model/model.pkl"
RANDOM_SEED = 42

def generate_fingerprints(smiles_list):
    """Generates Morgan fingerprints for a list of SMILES strings."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]

    # Filter out invalid molecules
    valid_mols = [mols[i] for i in valid_indices]

    # Generate fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in valid_mols]
    X = np.array(fps)

    return X, valid_indices

def train_model():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)

    print(f"Data shape: {df.shape}")

    # Extract features and targets
    smiles = df['canonical_smiles'].tolist()
    y = df['pchembl_value'].values

    print("Generating molecular fingerprints...")
    X, valid_indices = generate_fingerprints(smiles)

    # Filter y to match valid X
    y = y[valid_indices]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Train model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance (Test Set):")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train_model()
