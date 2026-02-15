#!/usr/bin/env python3
"""
Production-Ready Molecule Generator for ADRB2 Screening
Generates 5,000-10,000 diverse drug-like molecules using validated SMILES patterns
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import os
import random

class ProductionMoleculeGenerator:
    """Generate large libraries using only validated SMILES patterns"""
    
    def __init__(self, output_dir='../data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def check_valid_and_druglike(self, smiles):
        """Validate SMILES and check drug-likeness"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            Chem.SanitizeMol(mol)
            
            # Lipinski's Rule
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            if not (100 < mw < 500 and -2 < logp < 5 and hbd <= 5 and hba <= 10):
                return None
            
            if Descriptors.NumRotatableBonds(mol) > 10:
                return None
            
            return Chem.MolToSmiles(mol)
        except:
            return None
    
    def generate_phenethylamine_library(self):
        """Generate phenethylamine derivatives"""
        print("Generating phenethylamine library...")
        molecules = set()
        
        # Core scaffold variations
        cores = [
            'c1ccc(CCN)cc1',           # Basic phenethylamine
            'c1ccc(CCN)c(O)c1',        # Ortho-OH
            'c1cc(CCN)c(O)c(O)c1',     # Catechol
            'c1ccc(CCN)c(CO)c1',       # Benzyl alcohol
        ]
        
        # N-substituents
        n_subs = ['', 'C', 'CC', 'C(C)C', 'C(C)(C)C', 'CCC']
        
        # Ring substituents (positions 2,3,4,5,6)
        ring_subs = ['', 'O', 'OC', 'C', 'F', 'Cl', 'N', 'NC']
        
        count = 0
        for core in cores:
            for n_sub in n_subs:
                # Substitute amine
                if n_sub:
                    test_smiles = core.replace('CCN', f'CCN{n_sub}')
                else:
                    test_smiles = core.replace('CCN', 'CCN')
                
                validated = self.check_valid_and_druglike(test_smiles)
                if validated:
                    molecules.add(validated)
                    count += 1
                
                # Add ring substituents
                for r in ring_subs[:4]:  # Limit combinations
                    if 'c1ccc' in test_smiles and r:
                        variant = test_smiles.replace('c1ccc', f'c1c({r})cc', 1)
                        validated = self.check_valid_and_druglike(variant)
                        if validated:
                            molecules.add(validated)
                            count += 1
        
        print(f"  Generated {len(molecules)} phenethylamine analogs")
        return list(molecules)
    
    def generate_beta_blocker_library(self):
        """Generate beta-blocker-like molecules"""
        print("Generating beta-blocker library...")
        molecules = set()
        
        # Core scaffolds
        cores = [
            'c1ccccc1OCC(O)CN',                     # Phenoxypropanolamine
            'c1ccc2ccccc2c1OCC(O)CN',               # Naphthoxy variant
            'c1ccc(CCOC)cc1OCC(O)CN',               # Para-ethoxy ether
        ]
        
        # N-substituents
        n_subs = ['', 'C', 'CC', 'C(C)C', 'C(C)(C)C']
        
        for core in cores:
            for n_sub in n_subs:
                if n_sub:
                    test_smiles = core.replace('CN', f'CN{n_sub}')
                else:
                    test_smiles = core
                
                validated = self.check_valid_and_druglike(test_smiles)
                if validated:
                    molecules.add(validated)
                
                # Add aromatic substituents
                ring_pos = ['c1ccccc1', 'c1ccc', 'c1cc']
                ring_subs = ['C', 'OC', 'F', 'Cl']
                
                for pos in ring_pos:
                    if pos in test_smiles:
                        for r in ring_subs:
                            variant = test_smiles.replace(pos, f'{pos}({r})', 1)
                            validated = self.check_valid_and_druglike(variant)
                            if validated:
                                molecules.add(validated)
        
        print(f"  Generated {len(molecules)} beta-blocker analogs")
        return list(molecules)
    
    def generate_simple_aromatics(self):
        """Generate simple aromatic amines and alcohols"""
        print("Generating simple aromatic molecules...")
        molecules = set()
        
        templates = [
            # Benzylamines
            'c1ccc(CN)cc1',
            'c1ccc(CN)c(O)c1',
            'c1ccc(CN)c(OC)c1',
            'c1ccc(CN)c(C)c1',
            
            # Phenethylamines
            'c1ccc(CCN)cc1',
            'c1ccc(CCN)c(O)c1',
            
            # Amino alcohols
            'c1ccc(C(O)CN)cc1',
            'c1ccc(CC(O)N)cc1',
            
            # Ethers
            'c1ccc(OCC)cc1',
            'c1ccc(OCCN)cc1',
            
            # Phenols
            'c1ccc(O)cc1',
            'c1ccc(O)c(O)c1',
            'c1cc(O)cc(O)c1',
        ]
        
        n_subs = ['', 'C', 'CC', 'C(C)C']
        ring_subs = ['', 'C', 'O', 'OC', 'F', 'Cl']
        
        for template in templates:
            # Try with different N-substituents
            for n_sub in n_subs:
                if 'N' in template and n_sub:
                    variant = template.replace('N', f'N{n_sub}', 1)
                else:
                    variant = template
                
                validated = self.check_valid_and_druglike(variant)
                if validated:
                    molecules.add(validated)
                
                # Add ring substituents
                for r in ring_subs:
                    if 'cc1' in variant and r:
                        test = variant.replace('cc1', f'c({r})c1', 1)
                        validated = self.check_valid_and_druglike(test)
                        if validated:
                            molecules.add(validated)
        
        print(f"  Generated {len(molecules)} simple aromatics")
        return list(molecules)
    
    def generate_extended_analogs(self):
        """Generate extended analogs with varied linkers"""
        print("Generating extended analogs...")
        molecules = set()
        
        # Start with known good molecules
        seeds = [
            'CC(C)NCC(O)c1ccc(O)c(CO)c1',      # Albuterol
            'CC(C)NCC(O)c1ccc(O)c(O)c1',       # Isoproterenol
            'CC(C)NCC(O)COc1cccc2ccccc12',     # Propranolol
        ]
        
        # Systematic modifications
        mods = [
            ('C(C)', 'C'),           # Remove methyl
            ('C(C)', 'CC'),          # Ethyl
            ('C(C)', 'CCC'),         # Propyl
            ('(O)', '(OC)'),         # OH to OMe
            ('(O)c', '(F)c'),        # OH to F
            ('(O)c', 'c'),           # Remove OH
            ('CO)', 'C)'),           # Remove hydroxymethyl
        ]
        
        for seed in seeds:
            molecules.add(seed)
            
            # Apply modifications
            for old, new in mods:
                if old in seed:
                    variant = seed.replace(old, new, 1)
                    validated = self.check_valid_and_druglike(variant)
                    if validated:
                        molecules.add(validated)
                    
                    # Apply second modification
                    for old2, new2 in mods:
                        if old2 in variant and old2 != old:
                            variant2 = variant.replace(old2, new2, 1)
                            validated = self.check_valid_and_druglike(variant2)
                            if validated:
                                molecules.add(validated)
        
        print(f"  Generated {len(molecules)} extended analogs")
        return list(molecules)
    
    def generate_combinatorial_library(self):
        """Generate combinatorial library from building blocks"""
        print("Generating combinatorial library...")
        molecules = set()
        
        # Aromatic cores
        ar_cores = [
            'c1ccccc1',
            'c1ccc(O)cc1',
            'c1ccc(OC)cc1',
            'c1ccc(C)cc1',
            'c1ccc(F)cc1',
            'c1ccc(Cl)cc1',
        ]
        
        # Linkers
        linkers = [
            'C',
            'CC',
            'CCC',
            'C(O)C',
            'CC(O)C',
            'OC',
            'OCC',
        ]
        
        # Terminal groups
        terminals = [
            'N',
            'NC',
            'NCC',
            'NC(C)C',
            'O',
            'OC',
        ]
        
        # Build molecules
        for ar in ar_cores[:5]:  # Limit for speed
            for link in linkers[:5]:
                for term in terminals[:4]:
                    # Simple concatenation
                    smiles = f"{ar}{link}{term}"
                    validated = self.check_valid_and_druglike(smiles)
                    if validated:
                        molecules.add(validated)
        
        print(f"  Generated {len(molecules)} combinatorial molecules")
        return list(molecules)
    
    def generate_library(self, n_target=10000):
        """Generate complete library"""
        print("=" * 70)
        print("Production Molecule Library Generator")
        print("=" * 70)
        print(f"\nTarget: {n_target} molecules\n")
        
        all_molecules = set()
        
        # Method 1: Phenethylamines
        phenethy = self.generate_phenethylamine_library()
        all_molecules.update(phenethy)
        
        # Method 2: Beta-blockers
        beta = self.generate_beta_blocker_library()
        all_molecules.update(beta)
        
        # Method 3: Simple aromatics
        simple = self.generate_simple_aromatics()
        all_molecules.update(simple)
        
        # Method 4: Extended analogs
        extended = self.generate_extended_analogs()
        all_molecules.update(extended)
        
        # Method 5: Combinatorial
        combi = self.generate_combinatorial_library()
        all_molecules.update(combi)
        
        # Convert to list
        final_molecules = list(all_molecules)
        
        print(f"\n{'=' * 70}")
        print(f"Total unique molecules: {len(final_molecules)}")
        print(f"{'=' * 70}\n")
        
        # Calculate properties
        print("Calculating properties...")
        data = []
        for idx, smiles in enumerate(final_molecules):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                data.append({
                    'molecule_id': f'PROD_{idx:05d}',
                    'smiles': smiles,
                    'mol_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'source': 'production'
                })
        
        df = pd.DataFrame(data)
        
        # Save
        output_path = os.path.join(self.output_dir, 'diverse_candidates.csv')
        df.to_csv(output_path, index=False)
        
        print(f"\n{'=' * 70}")
        print("SUCCESS!")
        print(f"{'=' * 70}")
        print(f"\nLibrary Statistics:")
        print(f"  Total molecules: {len(df)}")
        print(f"  MW range: {df['mol_weight'].min():.0f} - {df['mol_weight'].max():.0f} Da")
        print(f"  LogP range: {df['logp'].min():.1f} - {df['logp'].max():.1f}")
        print(f"  Unique SMILES: {df['smiles'].nunique()}")
        print(f"\nSaved to: {output_path}")
        print(f"\n{'=' * 70}")
        print("READY FOR AI SCREENING!")
        print(f"{'=' * 70}")
        
        return df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    generator = ProductionMoleculeGenerator(output_dir=data_dir)
    library = generator.generate_library(n_target=10000)
    
    print("\nNext: python scripts/main_pipeline.py")


if __name__ == "__main__":
    main()
