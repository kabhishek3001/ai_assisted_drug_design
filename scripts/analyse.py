#!/usr/bin/env python3
"""
Analysis Script for ADRB2 Virtual Screening Results

Analyzes the predictions and generates insights about the screened molecules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_results(results_file='../results/ai_predictions.csv'):
    """Analyze virtual screening results"""
    
    print("=" * 70)
    print("ADRB2 Virtual Screening - Results Analysis")
    print("=" * 70)
    
    # Get script directory and construct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_path = os.path.join(project_dir, 'results', 'ai_predictions.csv')
    
    if not os.path.exists(results_path):
        print(f"\n✗ Results file not found: {results_path}")
        print("Run the pipeline first: python scripts/main_pipeline.py")
        return
    
    # Load results
    df = pd.read_csv(results_path)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total candidates analyzed: {len(df)}")
    print(f"  Unique molecules: {df['smiles'].nunique()}")
    
    # Prediction statistics
    print(f"\n🎯 Prediction Statistics:")
    print(f"  Min pIC50:    {df['predicted_pIC50'].min():.2f}")
    print(f"  Max pIC50:    {df['predicted_pIC50'].max():.2f}")
    print(f"  Mean pIC50:   {df['predicted_pIC50'].mean():.2f}")
    print(f"  Median pIC50: {df['predicted_pIC50'].median():.2f}")
    print(f"  Std Dev:      {df['predicted_pIC50'].std():.2f}")
    
    # Activity categories
    print(f"\n🏆 Activity Categories:")
    highly_potent = len(df[df['predicted_pIC50'] >= 8.0])
    potent = len(df[(df['predicted_pIC50'] >= 7.0) & (df['predicted_pIC50'] < 8.0)])
    moderate = len(df[(df['predicted_pIC50'] >= 6.0) & (df['predicted_pIC50'] < 7.0)])
    weak = len(df[df['predicted_pIC50'] < 6.0])
    
    print(f"  Highly Potent (pIC50 ≥ 8.0): {highly_potent} ({highly_potent/len(df)*100:.1f}%)")
    print(f"  Potent (7.0 ≤ pIC50 < 8.0):  {potent} ({potent/len(df)*100:.1f}%)")
    print(f"  Moderate (6.0 ≤ pIC50 < 7.0): {moderate} ({moderate/len(df)*100:.1f}%)")
    print(f"  Weak (pIC50 < 6.0):          {weak} ({weak/len(df)*100:.1f}%)")
    
    # Drug-likeness analysis
    print(f"\n💊 Drug-Likeness Properties:")
    print(f"  Molecular Weight:")
    print(f"    Range: {df['mol_weight'].min():.0f} - {df['mol_weight'].max():.0f} Da")
    print(f"    Mean:  {df['mol_weight'].mean():.0f} Da")
    
    print(f"  LogP (Lipophilicity):")
    print(f"    Range: {df['logp'].min():.2f} - {df['logp'].max():.2f}")
    print(f"    Mean:  {df['logp'].mean():.2f}")
    
    print(f"  H-Bond Donors:")
    print(f"    Range: {int(df['hbd'].min())} - {int(df['hbd'].max())}")
    print(f"    Mean:  {df['hbd'].mean():.1f}")
    
    print(f"  H-Bond Acceptors:")
    print(f"    Range: {int(df['hba'].min())} - {int(df['hba'].max())}")
    print(f"    Mean:  {df['hba'].mean():.1f}")
    
    # Top candidates
    print(f"\n🥇 Top 10 Candidates:")
    print("-" * 70)
    top10 = df.head(10)
    for idx, row in top10.iterrows():
        ic50_nm = 10 ** (9 - row['predicted_pIC50'])
        print(f"  {idx+1:2d}. {row['molecule_id']}: pIC50={row['predicted_pIC50']:.2f} (IC50≈{ic50_nm:.1f} nM)")
        print(f"      MW={row['mol_weight']:.0f} Da, LogP={row['logp']:.2f}")
        print(f"      SMILES: {row['smiles'][:60]}...")
        print()
    
    # Lipinski's Rule of Five compliance
    lipinski_pass = df[
        (df['mol_weight'] <= 500) &
        (df['logp'] <= 5) &
        (df['hbd'] <= 5) &
        (df['hba'] <= 10)
    ]
    
    print(f"\n✅ Lipinski's Rule of Five:")
    print(f"  Molecules passing: {len(lipinski_pass)}/{len(df)} ({len(lipinski_pass)/len(df)*100:.1f}%)")
    
    # Distribution analysis
    print(f"\n📈 Activity Distribution:")
    bins = [0, 5, 6, 7, 8, 9, 10]
    labels = ['<5', '5-6', '6-7', '7-8', '8-9', '9+']
    df['activity_bin'] = pd.cut(df['predicted_pIC50'], bins=bins, labels=labels)
    
    for label in labels:
        count = len(df[df['activity_bin'] == label])
        if count > 0:
            print(f"  pIC50 {label}: {count:3d} molecules ({count/len(df)*100:4.1f}%)")
    
    # Correlation analysis
    print(f"\n🔗 Property Correlations with Activity:")
    correlations = {
        'Molecular Weight': df['predicted_pIC50'].corr(df['mol_weight']),
        'LogP': df['predicted_pIC50'].corr(df['logp']),
        'H-Bond Donors': df['predicted_pIC50'].corr(df['hbd']),
        'H-Bond Acceptors': df['predicted_pIC50'].corr(df['hba'])
    }
    
    for prop, corr in correlations.items():
        direction = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        print(f"  {prop:20s}: {corr:+.3f} ({strength} {direction})")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if highly_potent > 0:
        print(f"  ✓ {highly_potent} highly potent candidates identified!")
        print(f"    → Prioritize these for experimental validation")
    
    if highly_potent < 5:
        print(f"  ⚠ Limited highly potent candidates found")
        print(f"    → Consider screening more molecules or adjusting filters")
    
    best_candidate = df.iloc[0]
    ic50_best = 10 ** (9 - best_candidate['predicted_pIC50'])
    
    print(f"\n  🎯 Best Candidate: {best_candidate['molecule_id']}")
    print(f"     Predicted IC50: {ic50_best:.2f} nM")
    print(f"     Next step: Molecular docking validation")
    
    print(f"\n{'=' * 70}")
    print("Analysis Complete!")
    print(f"{'=' * 70}")
    
    return df


def create_visualizations(df, output_dir='../results'):
    """Create visualization plots (optional - requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        plot_dir = os.path.join(project_dir, 'results', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot 1: Activity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['predicted_pIC50'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Predicted pIC50')
        plt.ylabel('Number of Molecules')
        plt.title('Distribution of Predicted Activities')
        plt.axvline(x=8.0, color='r', linestyle='--', label='Highly Potent (≥8.0)')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, 'activity_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"\n📊 Saved plot: {plot_dir}/activity_distribution.png")
        
        # Plot 2: MW vs Activity
        plt.figure(figsize=(10, 6))
        plt.scatter(df['mol_weight'], df['predicted_pIC50'], alpha=0.5)
        plt.xlabel('Molecular Weight (Da)')
        plt.ylabel('Predicted pIC50')
        plt.title('Molecular Weight vs Predicted Activity')
        plt.axhline(y=8.0, color='r', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(plot_dir, 'mw_vs_activity.png'), dpi=300, bbox_inches='tight')
        print(f"📊 Saved plot: {plot_dir}/mw_vs_activity.png")
        
        plt.close('all')
        
    except ImportError:
        print("\n(Matplotlib not available - skipping plots)")


def main():
    df = analyze_results()
    
    if df is not None:
        # Try to create visualizations
        try:
            create_visualizations(df)
        except Exception as e:
            print(f"\n(Could not create plots: {e})")
        
        print("\n📁 Files created:")
        print("  - results/ai_predictions.csv (screening results)")
        print("  - results/plots/*.png (visualizations, if matplotlib available)")
        
        print("\n🔍 Next steps:")
        print("  1. Review top candidates in ai_predictions.csv")
        print("  2. Look up molecules in ChEMBL database")
        print("  3. Set up molecular docking for validation")
        print("     → pip install meeko --break-system-packages")
        print("     → python scripts/main_pipeline.py")


if __name__ == "__main__":
    main()