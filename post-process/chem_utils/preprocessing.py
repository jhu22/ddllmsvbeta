import numpy as np
from rdkit import Chem

def remove_empty_strings(smiles_list):
    return np.array([smi for smi in smiles_list if smi != ''])

def remove_short_strings(smiles_list, min_length=3):
    return np.array([smi for smi in smiles_list if len(smi) >= min_length])

def get_unique_smiles(smiles_list):
    return np.unique(smiles_list)

def remove_invalid_molecules(smiles_list):
    return np.array([smi for smi in smiles_list if Chem.MolFromSmiles(smi) is not None])

def preprocess_smiles(smiles_list):
    cleaned = remove_short_strings(smiles_list)
    unique = get_unique_smiles(cleaned)
    valid = remove_invalid_molecules(unique)
    return valid