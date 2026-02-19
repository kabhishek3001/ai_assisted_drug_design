import pandas as pd
from chembl_webresource_client.new_client import new_client
import os

def fetch_adrb2_data():
    print("Connecting to ChEMBL database...")
    target = new_client.target
    activity = new_client.activity

    # UniProt ID for ADRB2 is P07550
    target_query = target.filter(target_components__accession='P07550').filter(target_type='SINGLE PROTEIN')
    targets = list(target_query)

    if not targets:
        print("Error: ADRB2 target not found in ChEMBL.")
        return

    selected_target = targets[0]['target_chembl_id']
    print(f"Found Target ChEMBL ID: {selected_target}")

    print("Fetching bioactivity data (IC50)...")
    res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

    df = pd.DataFrame.from_dict(res)

    if df.empty:
        print("No data found.")
        return

    print(f"Raw data shape: {df.shape}")

    # Filter for valid pChEMBL values and SMILES
    df = df[df.standard_value.notna()]
    df = df[df.canonical_smiles.notna()]
    df = df[df.pchembl_value.notna()]

    print(f"Filtered data shape (with pChEMBL and SMILES): {df.shape}")

    # Select relevant columns
    cols = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units', 'pchembl_value']
    df = df[cols]

    # Save to CSV
    output_dir = "ai_assisted_drug_design/data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "chembl_adrb2.csv")

    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    fetch_adrb2_data()
