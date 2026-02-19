import unittest
import os
import sys
import pandas as pd
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_assisted_drug_design.ai_model.predict import generate_fingerprints

class TestPipeline(unittest.TestCase):
    def test_fingerprint_generation(self):
        smiles = ["CCO", "c1ccccc1"]
        X, valid_indices = generate_fingerprints(smiles)
        self.assertEqual(len(valid_indices), 2)
        self.assertEqual(X.shape, (2, 2048))

    def test_model_exists(self):
        model_path = "ai_assisted_drug_design/ai_model/model.pkl"
        self.assertTrue(os.path.exists(model_path), "Model file should exist after training")

    def test_prediction_output(self):
        output_path = "ai_assisted_drug_design/results/predictions.csv"
        self.assertTrue(os.path.exists(output_path), "Prediction output file should exist")
        df = pd.read_csv(output_path)
        self.assertIn('predicted_pchembl_value', df.columns)
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()
