#!/usr/bin/env python3
"""
Visualization and Analysis Tools
Generate plots and reports for ADRB2 drug discovery results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class ResultsVisualizer:
    """
    Create visualizations for drug discovery results
    """
    
    def __init__(self, project_dir='../'):
        self.project_dir = project_dir
        self.results_dir = os.path.join(project_dir, 'results')
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set plotting style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def plot_model_performance(self, metrics, save=True):
        """
        Plot AI model training and test performance
        
        Args:
            metrics: Dictionary with train/test R² and RMSE
            save: Whether to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # R² scores
        r2_data = [metrics['train_r2'], metrics['test_r2']]
        axes[0].bar(['Training', 'Test'], r2_data, color=['#3498db', '#e74c3c'])
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Model Performance (R²)')
        axes[0].set_ylim([0, 1])
        axes[0].axhline(y=0.7, color='green', linestyle='--', label='Good threshold')
        axes[0].legend()
        
        # Add value labels
        for i, v in enumerate(r2_data):
            axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        # RMSE
        rmse_data = [metrics['train_rmse'], metrics['test_rmse']]
        axes[1].bar(['Training', 'Test'], rmse_data, color=['#3498db', '#e74c3c'])
        axes[1].set_ylabel('RMSE (pIC50 units)')
        axes[1].set_title('Prediction Error (RMSE)')
        
        for i, v in enumerate(rmse_data):
            axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.figures_dir, 'model_performance.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
    
    def plot_prediction_distribution(self, predictions_file='ai_predictions.csv', save=True):
        """
        Plot distribution of AI-predicted binding affinities
        
        Args:
            predictions_file: CSV with predictions
            save: Whether to save figure
        """
        predictions_path = os.path.join(self.results_dir, predictions_file)
        df = pd.read_csv(predictions_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Predicted pIC50 distribution
        axes[0, 0].hist(df['predicted_pIC50'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Predicted pIC50')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Predicted Activity')
        axes[0, 0].axvline(x=6.0, color='red', linestyle='--', label='Strong binder threshold')
        axes[0, 0].legend()
        
        # Molecular weight distribution
        axes[0, 1].hist(df['mol_weight'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Molecular Weight (Da)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Molecular Weight Distribution')
        axes[0, 1].axvline(x=500, color='red', linestyle='--', label="Lipinski's Rule")
        axes[0, 1].legend()
        
        # LogP distribution
        axes[1, 0].hist(df['logp'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('LogP')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Lipophilicity Distribution')
        axes[1, 0].axvline(x=5, color='red', linestyle='--', label="Lipinski's Rule")
        axes[1, 0].legend()
        
        # Top candidates table
        top_10 = df.nsmallest(10, 'predicted_pIC50', keep='first')[
            ['molecule_id', 'predicted_pIC50', 'mol_weight', 'logp']
        ]
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(
            cellText=top_10.values,
            colLabels=top_10.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        axes[1, 1].set_title('Top 10 Predicted Candidates', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.figures_dir, 'predictions_analysis.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
    
    def plot_docking_results(self, final_hits_file='final_hits.csv', save=True):
        """
        Plot docking results and AI-docking correlation
        
        Args:
            final_hits_file: CSV with combined results
            save: Whether to save figure
        """
        hits_path = os.path.join(self.results_dir, final_hits_file)
        
        if not os.path.exists(hits_path):
            print(f"Final hits file not found: {hits_path}")
            return
        
        df = pd.read_csv(hits_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # AI vs Docking scatter
        axes[0, 0].scatter(df['predicted_pIC50'], df['binding_affinity'], 
                          alpha=0.6, s=100, c=df['composite_score'], 
                          cmap='RdYlGn', edgecolors='black')
        axes[0, 0].set_xlabel('AI Predicted pIC50')
        axes[0, 0].set_ylabel('Docking Affinity (kcal/mol)')
        axes[0, 0].set_title('AI vs Physics-Based Validation')
        
        # Add quadrant lines
        axes[0, 0].axhline(y=-7.0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(x=6.0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].text(df['predicted_pIC50'].max() * 0.95, -7.5, 
                       'Strong binders', ha='right', fontweight='bold', color='green')
        
        cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
        cbar.set_label('Composite Score')
        
        # Binding affinity distribution
        axes[0, 1].hist(df['binding_affinity'], bins=20, color='#9b59b6', 
                       alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Binding Affinity (kcal/mol)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Docking Score Distribution')
        axes[0, 1].axvline(x=-7.0, color='red', linestyle='--', 
                          label='Strong binding threshold')
        axes[0, 1].legend()
        
        # Composite score distribution
        axes[1, 0].hist(df['composite_score'], bins=20, color='#1abc9c', 
                       alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Composite Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Overall Candidate Quality')
        
        # Top hits table
        top_hits = df.nlargest(10, 'composite_score')[
            ['molecule_id', 'predicted_pIC50', 'binding_affinity', 'composite_score']
        ].round(3)
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(
            cellText=top_hits.values,
            colLabels=top_hits.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)
        axes[1, 1].set_title('Top 10 Final Hits', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.figures_dir, 'docking_analysis.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
    
    def generate_report(self, final_hits_file='final_hits.csv'):
        """
        Generate summary report of top candidates
        
        Args:
            final_hits_file: CSV with final results
            
        Returns:
            Report text
        """
        hits_path = os.path.join(self.results_dir, final_hits_file)
        
        if not os.path.exists(hits_path):
            return "No final hits file found"
        
        df = pd.read_csv(hits_path)
        
        # Identify strong candidates
        strong_hits = df[
            (df['predicted_pIC50'] > 6.0) & 
            (df['binding_affinity'] < -7.0) &
            (df['composite_score'] > 0.7)
        ]
        
        report = f"""
{'=' * 70}
ADRB2 DRUG DISCOVERY - FINAL REPORT
{'=' * 70}

DATASET SUMMARY
---------------
Total candidates evaluated: {len(df)}
Candidates meeting strong hit criteria: {len(strong_hits)}

SELECTION CRITERIA
------------------
✓ AI Predicted pIC50 > 6.0 (strong binding predicted)
✓ Docking Affinity < -7.0 kcal/mol (favorable physical fit)
✓ Composite Score > 0.7 (top-tier overall)

TOP 5 CANDIDATES FOR EXPERIMENTAL VALIDATION
---------------------------------------------
"""
        
        top_5 = df.nlargest(5, 'composite_score')
        
        for idx, row in top_5.iterrows():
            report += f"""
Candidate: {row['molecule_id']}
  SMILES: {row['smiles']}
  AI Predicted pIC50: {row['predicted_pIC50']:.3f}
  Docking Affinity: {row['binding_affinity']:.3f} kcal/mol
  Molecular Weight: {row['mol_weight']:.1f} Da
  LogP: {row['logp']:.2f}
  Composite Score: {row['composite_score']:.3f}
  ---
"""
        
        report += f"""
DRUG-LIKENESS ASSESSMENT
------------------------
Lipinski's Rule of Five violations:
  MW > 500 Da: {len(df[df['mol_weight'] > 500])} molecules
  LogP > 5: {len(df[df['logp'] > 5])} molecules
  HBD > 5: {len(df[df['hbd'] > 5])} molecules
  HBA > 10: {len(df[df['hba'] > 10])} molecules

RECOMMENDATIONS
---------------
1. Synthesize top 3-5 candidates for in vitro testing
2. Perform functional assays on ADRB2-expressing cells
3. Evaluate selectivity vs ADRB1 and ADRB3
4. Conduct ADME profiling for lead optimization

{'=' * 70}
Report generated: {pd.Timestamp.now()}
{'=' * 70}
"""
        
        # Save report
        report_file = os.path.join(self.results_dir, 'final_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved: {report_file}")
        
        return report


def main():
    """
    Generate all visualizations and reports
    """
    print("=" * 60)
    print("Generating Visualizations and Reports")
    print("=" * 60)
    
    viz = ResultsVisualizer(project_dir='/home/claude/adrb2_discovery')
    
    # Check if we have predictions
    predictions_file = os.path.join(viz.results_dir, 'ai_predictions.csv')
    
    if os.path.exists(predictions_file):
        print("\n--- ANALYZING AI PREDICTIONS ---\n")
        viz.plot_prediction_distribution()
    
    # Check if we have final hits
    hits_file = os.path.join(viz.results_dir, 'final_hits.csv')
    
    if os.path.exists(hits_file):
        print("\n--- ANALYZING DOCKING RESULTS ---\n")
        viz.plot_docking_results()
        
        print("\n--- GENERATING FINAL REPORT ---\n")
        report = viz.generate_report()
        print(report)
    else:
        print("\nNo docking results found yet.")
        print("Run docking workflow first: python scripts/run_docking.py")
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print(f"Figures saved to: {viz.figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
