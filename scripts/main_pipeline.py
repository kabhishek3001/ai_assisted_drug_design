#!/usr/bin/env python3
"""
AI-Driven Discovery and Validation of ADRB2 Ligands
Main Pipeline Script

This script orchestrates the complete drug discovery workflow:
1. Data acquisition from ChEMBL
2. AI model training and validation
3. Virtual screening of candidate molecules
4. Integration with AutoDock Vina for physical validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ADRB2DrugDiscovery:
    """
    Main class for ADRB2 drug discovery pipeline
    """
    
    def __init__(self, project_dir='../'):
        self.project_dir = project_dir
        self.data_dir = os.path.join(project_dir, 'data')
        self.models_dir = os.path.join(project_dir, 'models')
        self.results_dir = os.path.join(project_dir, 'results')
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.model = None
        self.training_data = None
        
    def smiles_to_morgan_fingerprint(self, smiles, radius=2, n_bits=2048):
        """
        Convert SMILES string to Morgan fingerprint vector
        
        Args:
            smiles: SMILES string representation of molecule
            radius: Radius for Morgan fingerprint (default: 2)
            n_bits: Length of fingerprint bit vector (default: 2048)
            
        Returns:
            numpy array of fingerprint or None if conversion fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def load_and_prepare_data(self, chembl_file='chembl_adrb2_data.csv'):
        """
        Load ChEMBL data and prepare features
        
        Args:
            chembl_file: Path to ChEMBL dataset CSV
            
        Returns:
            DataFrame with features and activity values
        """
        filepath = os.path.join(self.data_dir, chembl_file)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"ChEMBL data file not found at {filepath}\n"
                f"Please run: python scripts/fetch_data.py"
            )
        
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Ensure required columns exist
        required_cols = ['canonical_smiles', 'pchembl_value']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}")
        
        # Remove rows with missing values
        df = df.dropna(subset=required_cols)
        
        # Convert pchembl_value to numeric
        df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
        df = df.dropna(subset=['pchembl_value'])
        
        print(f"Processing {len(df)} molecules...")
        
        # Generate Morgan fingerprints
        fingerprints = []
        valid_indices = []
        
        for idx, smiles in enumerate(df['canonical_smiles']):
            fp = self.smiles_to_morgan_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
        
        # Create feature matrix
        X = np.array(fingerprints)
        y = df.iloc[valid_indices]['pchembl_value'].values
        
        print(f"Successfully processed {len(X)} molecules")
        print(f"Feature matrix shape: {X.shape}")
        
        self.training_data = {
            'X': X,
            'y': y,
            'smiles': df.iloc[valid_indices]['canonical_smiles'].values
        }
        
        return self.training_data
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Train Random Forest model on prepared data
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        if self.training_data is None:
            raise ValueError("No training data loaded. Run load_and_prepare_data() first.")
        
        X = self.training_data['X']
        y = self.training_data['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print("\nTraining Random Forest Regressor...")
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
        }
        
        print("\n=== Model Performance ===")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, 'adrb2_rf_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to {model_path}")
        
        return metrics
    
    def virtual_screening(self, candidates_file='potential_candidates.csv', 
                         top_n=100, output_file='ai_predictions.csv'):
        """
        Screen candidate molecules using trained AI model
        
        Args:
            candidates_file: CSV file with candidate SMILES
            top_n: Number of top candidates to select
            output_file: Output filename for predictions
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            # Try to load saved model
            model_path = os.path.join(self.models_dir, 'adrb2_rf_model.pkl')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"Loaded model from {model_path}")
            else:
                raise ValueError("No trained model available. Run train_model() first.")
        
        # Load candidates
        candidates_path = os.path.join(self.data_dir, candidates_file)
        if not os.path.exists(candidates_path):
            raise FileNotFoundError(f"Candidates file not found at {candidates_path}")
        
        df = pd.read_csv(candidates_path)
        
        if 'smiles' not in df.columns and 'canonical_smiles' not in df.columns:
            raise ValueError("Candidates file must have 'smiles' or 'canonical_smiles' column")
        
        smiles_col = 'smiles' if 'smiles' in df.columns else 'canonical_smiles'
        
        print(f"\nScreening {len(df)} candidate molecules...")
        
        # Generate fingerprints
        fingerprints = []
        valid_indices = []
        
        for idx, smiles in enumerate(df[smiles_col]):
            fp = self.smiles_to_morgan_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
        
        X_candidates = np.array(fingerprints)
        
        # Predict
        predictions = self.model.predict(X_candidates)
        
        # Create results dataframe
        results = pd.DataFrame({
            'smiles': df.iloc[valid_indices][smiles_col].values,
            'predicted_pIC50': predictions,
            'molecule_id': df.iloc[valid_indices].get('molecule_id', range(len(valid_indices)))
        })
        
        # Add additional molecular properties
        print("Calculating molecular descriptors...")
        results['mol_weight'] = results['smiles'].apply(
            lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
        )
        results['logp'] = results['smiles'].apply(
            lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
        )
        results['hbd'] = results['smiles'].apply(
            lambda s: Descriptors.NumHDonors(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
        )
        results['hba'] = results['smiles'].apply(
            lambda s: Descriptors.NumHAcceptors(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
        )
        
        # Sort by predicted activity
        results = results.sort_values('predicted_pIC50', ascending=False)
        
        # Select top candidates
        top_candidates = results.head(top_n)
        
        # Save results
        output_path = os.path.join(self.results_dir, output_file)
        top_candidates.to_csv(output_path, index=False)
        
        print(f"\n=== Virtual Screening Complete ===")
        print(f"Processed: {len(X_candidates)} molecules")
        print(f"Top {top_n} candidates saved to {output_path}")
        print(f"\nTop 5 predictions:")
        print(top_candidates.head()[['molecule_id', 'predicted_pIC50', 'mol_weight', 'logp']])
        
        return top_candidates
    
    def prepare_for_docking(self, predictions_file='ai_predictions.csv', 
                           top_n=10, output_dir='ligands'):
        """
        Prepare top candidates for AutoDock Vina docking
        
        Args:
            predictions_file: CSV with AI predictions
            top_n: Number of top molecules to prepare
            output_dir: Directory for ligand files
            
        Returns:
            List of prepared ligand files
        """
        try:
            from meeko import MoleculePreparation
            from meeko import PDBQTWriterLegacy
        except ImportError:
            print("Warning: Meeko not installed. Install with: pip install meeko --break-system-packages")
            print("Skipping ligand preparation...")
            return []
        
        predictions_path = os.path.join(self.results_dir, predictions_file)
        df = pd.read_csv(predictions_path)
        
        # Take top N
        top_mols = df.head(top_n)
        
        ligands_dir = os.path.join(self.project_dir, 'docking', output_dir)
        os.makedirs(ligands_dir, exist_ok=True)
        
        prepared_files = []
        
        print(f"\nPreparing {len(top_mols)} ligands for docking...")
        
        for idx, row in top_mols.iterrows():
            mol_id = row.get('molecule_id', idx)
            smiles = row['smiles']
            
            try:
                # Convert SMILES to 3D
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                
                # Prepare for docking
                preparator = MoleculePreparation()
                preparator.prepare(mol)
                
                # Write PDBQT
                # For meeko >= 0.5, we should use preparator.write_pdbqt_string()
                pdbqt_string = preparator.write_pdbqt_string()
                
                output_file = os.path.join(ligands_dir, f'ligand_{mol_id}.pdbqt')
                with open(output_file, 'w') as f:
                    f.write(pdbqt_string)
                
                prepared_files.append(output_file)
                print(f"  Prepared: ligand_{mol_id}.pdbqt")
                
            except Exception as e:
                print(f"  Error preparing molecule {mol_id}: {e}")
                continue
        
        print(f"\nSuccessfully prepared {len(prepared_files)} ligands")
        return prepared_files


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("AI-Driven Discovery and Validation of ADRB2 Ligands")
    print("=" * 60)
    
    # Get the script directory and construct project path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    print(f"\nProject directory: {project_dir}\n")
    
    # Initialize pipeline
    pipeline = ADRB2DrugDiscovery(project_dir=project_dir)
    
    # Phase 1: Load data and train model
    print("\n--- PHASE 1: AI MODEL TRAINING ---\n")
    try:
        pipeline.load_and_prepare_data()
        metrics = pipeline.train_model()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo proceed, you need to:")
        print("1. Run: python scripts/fetch_data.py")
        print("   This will download ADRB2 data from ChEMBL")
        return
    
    # Phase 2: Virtual screening
    print("\n--- PHASE 2: VIRTUAL SCREENING ---\n")
    try:
        top_candidates = pipeline.virtual_screening(top_n=100)
    except FileNotFoundError:
        print("Candidates file not found.")
        print("Run: python scripts/fetch_data.py")
        return
    
    # Phase 3: Prepare for docking
    print("\n--- PHASE 3: DOCKING PREPARATION ---\n")
    prepared = pipeline.prepare_for_docking(top_n=10)
    
    if prepared:
        print("\n=== NEXT STEPS ===")
        print("1. Ensure receptor file is ready: protein/beta2_receptor.pdb")
        print("2. Configure docking box in: docking/vina_config.txt")
        print("3. Run docking with: python scripts/run_docking.py")
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()