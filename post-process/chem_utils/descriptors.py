import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from chem_utils.utils import cal_SA
from chem_utils.utils.cal_SA import sascorer

def smiles_to_mol_list(smiles_list):
    mol_list = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Invalid SMILES at index {i}")
        else:
            mol_list.append(mol)
    return mol_list

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def compute_qed_from_smiles(smiles_list):
    qed_scores = []
    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        qed = QED.qed(mol) if mol else 0
        if mol is None:
            print(f"Invalid SMILES at index {i}")
        qed_scores.append(qed)
    return np.array(qed_scores)

def compute_qed_from_mols(mol_list):
    return np.array([QED.qed(mol) for mol in mol_list])

def compute_sa_from_mols(mol_list):
    return np.array([sascorer.calculateScore(mol) for mol in mol_list])

def compute_sa_from_smiles(smiles_list):
    sa_scores = []
    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        sa = sascorer.calculateScore(mol) if mol else 0
        if mol is None:
            print(f"Invalid SMILES at index {i}")
        sa_scores.append(sa)
    return np.array(sa_scores)
