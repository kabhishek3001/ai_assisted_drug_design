import unittest
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestVirtualScreening(unittest.TestCase):
    def test_pipeline_output(self):
        output_dir = "ai_assisted_drug_design/results/screening_run"
        final_report = os.path.join(output_dir, "final_screening_report.csv")
        self.assertTrue(os.path.exists(final_report), "Final screening report should exist")

        df = pd.read_csv(final_report)
        self.assertIn('ai_pchembl_value', df.columns)
        self.assertIn('docking_score_kcal_mol', df.columns)
        self.assertGreater(len(df), 0, "Report should contain results")

        # Verify sorting (best docking score first - most negative)
        scores = df['docking_score_kcal_mol'].tolist()
        self.assertEqual(scores, sorted(scores), "Report should be sorted by docking score (ascending - most negative first)")

    def test_plot_exists(self):
        plot_path = "ai_assisted_drug_design/results/plots/correlation.png"
        self.assertTrue(os.path.exists(plot_path), "Correlation plot should be generated")

if __name__ == "__main__":
    unittest.main()
