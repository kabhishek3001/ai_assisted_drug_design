#!/usr/bin/env python3
"""
Convert downloaded molecules to the format needed for screening
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import sys

def convert_downloaded_to_candidates(
    input_file, 
    output_file='diverse_candidates.csv',
    max_molecules=10000
):
    """
    Convert downloaded molecules to screening format
    
    Args:
        input_file: Path to downloaded file
        output_file: Where to save converted file
        max_molecules: Maximum number to process
    """
    
    print("=" * 70)
    print("Converting Downloaded Molecules")
    print("=" * 70)
    print(f"\nInput file: {input_file}")
    print(f"Max molecules: {max_molecules}\n")
    
    # Try to read the file
    molecules = []
    smiles_list = []
    
    # Detect format and read
    try:
        print("Attempting to read as TSV/CSV...")
        # Try reading as tab-separated first
        df_raw = pd.read_csv(input_file, sep='\t', nrows=1)
        
        # Check if it worked
        if len(df_raw.columns) > 1:
            print("  ✓ Detected tab-separated format")
            df_raw = pd.read_csv(input_file, sep='\t')
        else:
            # Try comma-separated
            df_raw = pd.read_csv(input_file, sep=',')
            print("  ✓ Detected comma-separated format")
        
        # Find SMILES column
        smiles_col = None
        possible_names = ['smiles', 'SMILES', 'canonical_smiles', 'Smiles']
        
        for col in df_raw.columns:
            if col in possible_names or 'smiles' in col.lower():
                smiles_col = col
                break
        
        if smiles_col is None:
            # First column might be SMILES
            smiles_col = df_raw.columns[0]
            print(f"  ! No SMILES column found, using first column: {smiles_col}")
        else:
            print(f"  ✓ Found SMILES column: {smiles_col}")
        
        smiles_list = df_raw[smiles_col].dropna().tolist()[:max_molecules]
        
    except Exception as e:
        # Try reading as simple text file (one SMILES per line)
        print("  Tab/CSV format failed, trying simple text format...")
        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()
                # Skip header if present
                if lines and not any(c in lines[0] for c in ['c', 'C', 'N', 'O']):
                    lines = lines[1:]
                smiles_list = [line.strip().split()[0] for line in lines if line.strip()][:max_molecules]
            print("  ✓ Detected simple text format (one SMILES per line)")
        except Exception as e2:
            print(f"  ✗ Error reading file: {e2}")
            return None
    
    print(f"\nFound {len(smiles_list)} SMILES strings to process")
    
    # Validate and filter
    print("\nValidating molecules...")
    valid_data = []
    errors = 0
    
    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                errors += 1
                continue
            
            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Filter by Lipinski's Rule of Five
            if not (100 < mw < 500):
                continue
            if not (-2 < logp < 5):
                continue
            if hbd > 5 or hba > 10:
                continue
            
            # Get canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol)
            
            valid_data.append({
                'molecule_id': f'DL_{idx:05d}',
                'smiles': canonical_smiles,
                'mol_weight': mw,
                'logp': logp,
                'hbd': hbd,
                'hba': hba,
                'source': 'downloaded'
            })
            
            if len(valid_data) % 1000 == 0:
                print(f"  Processed {len(valid_data)} valid molecules...")
        
        except Exception as e:
            errors += 1
            continue
    
    print(f"\nValidation complete:")
    print(f"  Valid molecules: {len(valid_data)}")
    print(f"  Invalid/filtered: {len(smiles_list) - len(valid_data)}")
    
    if len(valid_data) == 0:
        print("\n✗ No valid molecules found!")
        print("  Check that your file contains valid SMILES strings")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(valid_data)
    
    # Remove duplicates
    original_count = len(df)
    df = df.drop_duplicates(subset=['smiles'])
    
    if original_count != len(df):
        print(f"  Removed {original_count - len(df)} duplicates")
    
    print(f"\nFinal library: {len(df)} unique molecules")
    
    # Determine output path
    if os.path.isabs(output_file):
        output_path = output_file
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        output_path = os.path.join(project_dir, 'data', output_file)
    
    # Save
    df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 70}")
    print("SUCCESS!")
    print(f"{'=' * 70}")
    print(f"\nSaved to: {output_path}")
    print(f"\nLibrary Statistics:")
    print(f"  Molecules: {len(df)}")
    print(f"  MW range: {df['mol_weight'].min():.0f} - {df['mol_weight'].max():.0f} Da")
    print(f"  LogP range: {df['logp'].min():.1f} - {df['logp'].max():.1f}")
    print(f"  HBD range: {int(df['hbd'].min())} - {int(df['hbd'].max())}")
    print(f"  HBA range: {int(df['hba'].min())} - {int(df['hba'].max())}")
    
    return df


def main():
    if len(sys.argv) < 2:
        print("=" * 70)
        print("Convert Downloaded Molecules to Screening Format")
        print("=" * 70)
        print("\nUsage: python convert_downloaded.py <input_file> [max_molecules]")
        print("\nExamples:")
        print("  python convert_downloaded.py ../data/zinc_druglike.txt")
        print("  python convert_downloaded.py ../data/chembl_molecules.txt 10000")
        print("  python convert_downloaded.py ../data/downloaded.smi 5000")
        print("\nSupported formats:")
        print("  - Tab-separated (TSV)")
        print("  - Comma-separated (CSV)")
        print("  - Simple text (one SMILES per line)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    max_mols = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"✗ Error: File not found: {input_file}")
        print(f"\nMake sure you:")
        print(f"  1. Downloaded the molecules file")
        print(f"  2. Saved it to the correct location")
        print(f"  3. Used the correct path")
        sys.exit(1)
    
    result = convert_downloaded_to_candidates(input_file, max_molecules=max_mols)
    
    if result is not None:
        print("\n" + "=" * 70)
        print("READY FOR SCREENING!")
        print("=" * 70)
        print("\nNext step: python scripts/main_pipeline.py")
    else:
        print("\n✗ Conversion failed. Please check your input file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
