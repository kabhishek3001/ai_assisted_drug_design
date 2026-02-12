#!/bin/bash

#README.md for AI Assisted Drug Design Project

# AI Assisted Drug Design for Asthma (β2‑Adrenergic Receptor Target)

## Overview

This project builds an AI‑assisted computational drug discovery pipeline targeting the Beta‑2 Adrenergic Receptor (ADRB2), a key protein involved in asthma treatment.

The pipeline combines:

- Artificial Intelligence (Machine Learning)
- Molecular Docking (AutoDock Vina)
- Protein Structure Analysis (AlphaFold)
- Ligand Dataset Analysis (ChEMBL / BindingDB)

This project is designed for academic research and conference presentation.

---

## Target Protein

Protein Name: Beta‑2 Adrenergic Receptor  
Gene Name: ADRB2  
UniProt ID: P07550  
Disease: Asthma  

Function: Activation of this receptor relaxes airway muscles and improves breathing.

Protein structure obtained from AlphaFold Protein Structure Database.

---

## Project Workflow

Step 1: Collect protein structure (AlphaFold / UniProt)

Step 2: Collect ligand dataset (ChEMBL / BindingDB)

Step 3: Preprocess dataset

Step 4: Extract molecular features using RDKit

Step 5: Train AI model to predict binding affinity

Step 6: Predict new candidate molecules

Step 7: Perform molecular docking using AutoDock Vina

Step 8: Rank molecules based on docking score and AI prediction

---

## Repository Structure

ai_assisted_drug_design/

data/  
    raw/  
    processed/  

protein/  
    beta2_receptor.pdb  

ligands/  
    sdf/  
    pdbqt/  

ai_model/  
    train.py  
    predict.py  
    model.pkl  

docking/  
    vina_config.txt  
    results/  

results/  
    final_hits.csv  

notebooks/  
    exploration.ipynb  

requirements.txt  
README.md  

---

## Role of Artificial Intelligence

The AI model learns from known drug molecules and predicts binding affinity of new molecules.

Benefits:

- Faster screening
- Reduced computation
- Intelligent candidate selection

Models used may include:

- Random Forest
- Neural Network
- Gradient Boosting

---

## Role of AutoDock Vina

AutoDock Vina performs molecular docking to:

- Simulate ligand‑protein interaction
- Calculate binding affinity score
- Identify best drug candidates

Output:

Docking Score (kcal/mol)

Lower score indicates stronger binding.

---

## Installation

Clone the repository:

git clone https://github.com/kabhishek3001/ai_assisted_drug_design.git

cd ai_assisted_drug_design

Install dependencies:

pip install -r requirements.txt

---

## Usage

Train AI model:

python ai_model/train.py

Predict new molecules:

python ai_model/predict.py

Run docking:

vina --config docking/vina_config.txt

---

## Input Files

Protein structure:

protein/beta2_receptor.pdb

Dataset:

data/raw/

Ligands:

ligands/sdf/

---

## Output Files

Trained model:

ai_model/model.pkl

Docking results:

docking/results/

Final candidates:

results/final_hits.csv

---

## Applications

- AI‑assisted drug discovery
- Asthma drug research
- Computational biology
- Academic and conference projects

---

## Disclaimer

This project is for research and educational purposes only. Not intended for clinical use.

---

## Author

Abhishek  
GitHub: https://github.com/kabhishek3001
