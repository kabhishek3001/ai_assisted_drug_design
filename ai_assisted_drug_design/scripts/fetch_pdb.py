import requests
import os

def fetch_pdb(pdb_id, output_dir):
    """Downloads a PDB file from RCSB."""
    print(f"Fetching PDB structure {pdb_id}...")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    response = requests.get(url)

    if response.status_code == 200:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "beta2_receptor.pdb")
        with open(output_path, "w") as f:
            f.write(response.text)
        print(f"Successfully saved to {output_path}")
    else:
        print(f"Error: Failed to download PDB file (Status Code: {response.status_code})")

if __name__ == "__main__":
    fetch_pdb("2RH1", "ai_assisted_drug_design/protein")
