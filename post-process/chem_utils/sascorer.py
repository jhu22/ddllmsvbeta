import math
import gzip
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def load_fragment_scores(path='./fpscores.pkl.gz'):
    """Load precomputed fragment scores from a gzipped pickle file."""
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    fscores = {}
    for entry in data:
        score = float(entry[0])
        for bit_id in entry[1:]:
            fscores[bit_id] = score
    return fscores

def get_fragment_score(mol, fscores, radius=2, unknown_penalty=-4.0):
    """Compute the fragment contribution score from a circular fingerprint."""
    fp = rdMolDescriptors.GetMorganFingerprint(mol, radius)
    fps = fp.GetNonzeroElements()
    total_score = 0.0
    total_count = 0
    for bit_id, count in fps.items():
        score = fscores.get(bit_id, unknown_penalty)
        total_score += score * count
        total_count += count
    avg_score = total_score / total_count if total_count else unknown_penalty
    return avg_score, len(fps)


def compute_features_penalty(mol):
    """Compute penalties for size, stereochemistry, spiro/bridge atoms, macrocycles."""
    n_atoms = mol.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    n_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_macrocycles = sum(1 for ring in ri.AtomRings() if len(ring) > 8)

    size_penalty = n_atoms**1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

    return - (size_penalty + stereo_penalty + spiro_penalty + bridge_penalty + macrocycle_penalty)


def compute_symmetry_correction(n_atoms, n_fps):
    """Apply a correction factor for highly symmetrical molecules."""
    if n_atoms > n_fps:
        return math.log(float(n_atoms) / n_fps) * 0.5
    return 0.0


def normalize_score(raw_score, min_val=-4.0, max_val=2.5):
    """Normalize the raw score to a 1–10 scale with upper-end smoothing."""
    sascore = 11.0 - ((raw_score - min_val + 1.0) / (max_val - min_val)) * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    sascore = min(max(sascore, 1.0), 10.0)
    return sascore


def calculate_score(mol, fscores):
    """Calculate the full synthetic accessibility (SA) score."""
    frag_score, n_fps = get_fragment_score(mol, fscores)
    feature_penalty = compute_features_penalty(mol)
    symmetry_correction = compute_symmetry_correction(mol.GetNumAtoms(), n_fps)
    raw_score = frag_score + feature_penalty + symmetry_correction
    return normalize_score(raw_score)