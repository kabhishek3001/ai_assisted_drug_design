#!/usr/bin/env python3
"""
ChEMBL Data Fetcher for ADRB2
Downloads bioactivity data for Beta-2 Adrenergic Receptor (Target P07550)
"""

import pandas as pd
import requests
import time
import os

def fetch_chembl_data(target_uniprot='P07550', output_dir='../data'):
    """
    Fetch bioactivity data from ChEMBL database
    
    Args:
        target_uniprot: UniProt ID for ADRB2 (P07550)
        output_dir: Directory to save data
        
    Returns:
        DataFrame with bioactivity data
    """
    print(f"Fetching data for ADRB2 (UniProt: {target_uniprot})")
    
    # ChEMBL REST API endpoints
    base_url = "https://www.ebi.ac.uk/chembl/api/data"
    
    try:
        # Step 1: Get target ChEMBL ID from UniProt
        print("\nStep 1: Getting ChEMBL target ID...")
        target_url = f"{base_url}/target.json?target_components__accession={target_uniprot}"
        
        response = requests.get(target_url)
        response.raise_for_status()
        target_data = response.json()
        
        if not target_data['targets']:
            raise ValueError(f"No target found for UniProt ID {target_uniprot}")
        
        target_chembl_id = target_data['targets'][0]['target_chembl_id']
        print(f"  Found target: {target_chembl_id}")
        
        # Step 2: Get bioactivities
        print("\nStep 2: Fetching bioactivity data...")
        activities_url = f"{base_url}/activity.json"
        
        all_activities = []
        offset = 0
        limit = 1000
        
        while True:
            params = {
                'target_chembl_id': target_chembl_id,
                'limit': limit,
                'offset': offset,
                'standard_type__in': 'IC50,EC50,Ki,Kd',  # Common activity types
                'pchembl_value__isnull': False  # Must have pChEMBL value
            }
            
            response = requests.get(activities_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            activities = data.get('activities', [])
            all_activities.extend(activities)
            
            print(f"  Retrieved {len(all_activities)} activities...")
            
            # Check if we got all data
            if len(activities) < limit:
                break
            
            offset += limit
            time.sleep(0.5)  # Be nice to the API
        
        print(f"\nTotal activities retrieved: {len(all_activities)}")
        
        # Step 3: Process and structure data
        print("\nStep 3: Processing data...")
        
        processed_data = []
        for activity in all_activities:
            # Extract relevant fields
            row = {
                'molecule_chembl_id': activity.get('molecule_chembl_id'),
                'canonical_smiles': activity.get('canonical_smiles'),
                'standard_type': activity.get('standard_type'),
                'standard_value': activity.get('standard_value'),
                'standard_units': activity.get('standard_units'),
                'pchembl_value': activity.get('pchembl_value'),
                'assay_chembl_id': activity.get('assay_chembl_id'),
                'assay_description': activity.get('assay_description'),
                'assay_type': activity.get('assay_type'),
                'target_organism': activity.get('target_organism')
            }
            processed_data.append(row)
        
        df = pd.DataFrame(processed_data)
        
        # Remove duplicates and missing SMILES
        df = df.dropna(subset=['canonical_smiles', 'pchembl_value'])
        df = df.drop_duplicates(subset=['canonical_smiles'])
        
        print(f"Processed dataset size: {len(df)} unique molecules")
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'chembl_adrb2_data.csv')
        df.to_csv(output_file, index=False)
        
        print(f"\n=== Data Download Complete ===")
        print(f"Saved to: {output_file}")
        print(f"\nDataset summary:")
        print(f"  Total molecules: {len(df)}")
        print(f"  pChEMBL range: {df['pchembl_value'].min():.2f} - {df['pchembl_value'].max():.2f}")
        print(f"  Mean pChEMBL: {df['pchembl_value'].mean():.2f}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ChEMBL: {e}")
        print("\nAlternative: Download manually from:")
        print(f"https://www.ebi.ac.uk/chembl/target_report_card/{target_uniprot}/")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def create_example_candidates(output_dir='../data', n_molecules=500):
    """
    Create example candidate molecules dataset
    (In practice, these would come from a virtual library or vendor catalog)
    
    Args:
        output_dir: Directory to save file
        n_molecules: Number of example molecules
    """
    print(f"\nCreating example candidates dataset ({n_molecules} molecules)...")
    
    # Example SMILES from various drug-like molecules
    # In reality, you'd use a large virtual screening library
    example_smiles = [
        'CC(C)NCC(O)c1ccc(O)c(O)c1',  # Isoproterenol (known ADRB2 agonist)
        'CC(C)NCC(O)c1ccc(O)cc1',  # Isoprenaline
        'CCC(C)NCC(O)c1ccc(O)c(O)c1',  # Similar structure
        'CC(C)NCC(O)c1ccc(O)c(CO)c1',  # Variant
        'CCNCC(O)c1ccc(O)c(O)c1',  # Another variant
    ]
    
    # Generate variations (this is just for demonstration)
    molecules = []
    for i in range(n_molecules):
        if i < len(example_smiles):
            smiles = example_smiles[i]
        else:
            # Repeat with modifications (just for example)
            smiles = example_smiles[i % len(example_smiles)]
        
        molecules.append({
            'molecule_id': f'CAND_{i:04d}',
            'smiles': smiles,
            'source': 'virtual_library'
        })
    
    df = pd.DataFrame(molecules)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'potential_candidates.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Example candidates saved to: {output_file}")
    print(f"Note: Replace with real virtual screening library for production use")
    
    return df


def main():
    """
    Main data fetching workflow
    """
    print("=" * 60)
    print("ChEMBL Data Fetcher for ADRB2")
    print("=" * 60)
    
    # Get the script directory and construct data path relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    print(f"\nProject directory: {project_dir}")
    print(f"Data directory: {data_dir}\n")
    
    # Fetch training data
    print("\n--- FETCHING TRAINING DATA ---\n")
    training_data = fetch_chembl_data(
        target_uniprot='P07550',
        output_dir=data_dir
    )
    
    # Create example candidates
    print("\n--- CREATING EXAMPLE CANDIDATES ---\n")
    candidates = create_example_candidates(
        output_dir=data_dir,
        n_molecules=1000
    )
    
    if training_data is not None and candidates is not None:
        print("\n" + "=" * 60)
        print("Data Preparation Complete!")
        print("=" * 60)
        print("\nNext step: Run the main pipeline")
        print("  python scripts/main_pipeline.py")


if __name__ == "__main__":
    main()