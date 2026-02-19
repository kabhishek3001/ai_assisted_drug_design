import unittest
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestDockingPipeline(unittest.TestCase):
    def test_receptor_exists(self):
        receptor_path = "ai_assisted_drug_design/protein/prepared/beta2_receptor.pdbqt"
        self.assertTrue(os.path.exists(receptor_path), "Prepared receptor file should exist")

    def test_ligands_prepared(self):
        ligand_dir = "ai_assisted_drug_design/ligands/pdbqt"
        files = os.listdir(ligand_dir)
        pdbqt_files = [f for f in files if f.endswith(".pdbqt")]
        self.assertGreater(len(pdbqt_files), 0, "Should have prepared ligand PDBQT files")
        self.assertIn("Propranolol.pdbqt", pdbqt_files)

    def test_docking_results(self):
        summary_path = "ai_assisted_drug_design/docking/results/docking_summary.csv"
        self.assertTrue(os.path.exists(summary_path), "Docking summary file should exist")
        df = pd.read_csv(summary_path)
        self.assertIn('docking_score', df.columns)
        self.assertGreater(len(df), 0)

        # Check if Propranolol has a reasonable score (better than -6.0)
        propranolol = df[df['molecule_name'] == 'Propranolol']
        if not propranolol.empty:
            score = propranolol.iloc[0]['docking_score']
            self.assertLess(score, -6.0, "Propranolol should have a strong binding affinity (< -6.0)")

if __name__ == "__main__":
    unittest.main()
