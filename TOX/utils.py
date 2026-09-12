from collections import defaultdict
import random

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MolToSmiles
import pandas as pd

import os

def one_hot_encoding(x, permitted_list):
    """
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    Parameters
    ----------
    x : str
        Element to be encoded.
    permitted_list : list
        List of permitted elements.
    Returns
    -------
    list
        One-hot encoded vector.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    
    Parameters
    ----------
    atom : RDKit atom object
        Atom to be encoded.
    use_chirality : bool
        Whether to use chirality features.
    hydrogens_implicit : bool
        Whether to use implicit hydrogens.
    Returns
    -------
    np.array
        Atom feature vector.
    """
    # list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Takes an RDKit bond object as input and gives a 1d array of bond features as output.
    
    Parameters
    ----------
    bond : RDKit bond object
        Bond to be encoded.
    use_stereochemistry : bool
        Whether to use stereochemistry features.
    Returns
    -------
    np.array
        Bond feature vector.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)


def smiles_to_graph_list(x_smiles, y):
    """ 
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Creates a list of PyTorch Geometric Data objects from smiles strings and labels.
    
    Parameters
    ----------
    x_smiles : list
        List of smiles strings.
   y : list
       List of labels.
   Returns
   -------
   list
       List of PyTorch Geometric Data objects.
   """
    data_list = []
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

    for smiles, y_val in zip(x_smiles, y):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()

        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
        X = torch.tensor(X, dtype=torch.float)

        rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        EF = np.zeros((n_edges, n_edge_features))
        for k, (i, j) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
        EF = torch.tensor(EF, dtype=torch.float)

        y_tensor = torch.tensor([y_val], dtype=torch.float)
        data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor))

    return data_list


def save_ckp(state, is_best, task_ckp_dir,
              best_model_dir, model_fname, 
              best_model_fname):
    """
    Saves the checkpoint to a file.
    
    Parameters
    ----------
    state : dict
        Checkpoint state.
    is_best : bool
        Whether the checkpoint is the best.
    checkpoint_dir : str
        Checkpoint directory.
    best_model_dir : str
        Best model directory.
    filename : str
        Filename.
    best_model : str
        Best model name.
    Returns
    -------
    None
    """
    f_path = os.path.join(task_ckp_dir, model_fname)
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, best_model_fname)
        torch.save(state, best_fpath)
        if hasattr(os, 'sync'):
            os.sync()


def optimizer_to(optim, device):
    """
    Moves the optimizer to the specified device.
    
    Parameters
    ----------
    optim : torch.optim.Optimizer
        Optimizer to move.
    device : torch.device
        Device to move the optimizer to.
    Returns
    -------
    None
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    Loads the checkpoint from a file.
    
    Parameters
    ----------
    checkpoint_fpath : str
        Checkpoint file path.
    model : torch.nn.Module
        Model to load the checkpoint to.
    optimizer : torch.optim.Optimizer
        Optimizer to load the checkpoint to.
    Returns
    -------
    tuple
        Tuple of (model, optimizer, epoch).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fpath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
    optimizer_to(optimizer, device)
    
    return model, optimizer, checkpoint['epoch']

def is_valid_smiles(smiles: str) -> bool:
    """
    Checks if a SMILES string is valid.
    
    Parameters
    ----------
    smiles : str
        SMILES string to check.
    Returns
    -------
    bool
        True if the SMILES string is valid, False otherwise.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        print(f'Exception: {e}\n smiles :{smiles}')
        return False

from typing import List

def get_valid_mask(smiles_series: pd.Series) -> pd.Series:
    """
    Returns a boolean Series mask indicating valid SMILES.
    """
    total_len =len(smiles_series)
    smiles_df = smiles_series.apply(is_valid_smiles)
    valid_len = len(smiles_df)
    print(f"Total # SMILES/Valid # SMILES: {total_len/valid_len}")
    return smiles_df

def get_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Cant convert to Mol: {smiles}\n")
            return "INVALID"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        if scaffold == "":
            # Acyclic molecule — use canonical SMILES as its own group key
            return Chem.MolToSmiles(mol, canonical=True)
        return scaffold
    except Exception:
        return "INVALID"


def scaffold_split(smiles_list:List[str], train_frac:float, val_frac:float, seed:int):
    """
    Split a list of SMILES strings into train, validation, and test sets based on Bemis-Murcko scaffolds.
    The split is done such that molecules with the same scaffold are in the same set (includeChirality=True).
 
    Parameters
    ----------
    smiles_list : list
        List of smiles strings.
    train_frac : float
        Fraction of data to use for training.
    val_frac : float
        Fraction of data to use for validation.
    seed : int
        Random seed.
    Returns
    -------
    tuple
        Tuple of (train_idx, val_idx, test_idx).
    """
    # Map scaffold → list of molecule indices
    scaffolds = {}
    for idx, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(idx) 
    # Sort scaffold groups by size (largest first)
    scaffold_groups = sorted(
        scaffolds.values(), key=lambda x: len(x), reverse=True
    )
 
    n_total = len(smiles_list)
    train_cutoff = int(train_frac * n_total)
    val_cutoff   = int((train_frac + val_frac) * n_total)
 
    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) < train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) < val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
 
    print(
      f"Scaffold split → train: {len(train_idx)} ({len(train_idx)/n_total:.2%}),"
      f" val: {len(val_idx)} ({len(val_idx)/n_total:.2%}), test:"
      f" {len(test_idx)} ({len(test_idx)/n_total:.2%})"
  )
    return train_idx, val_idx, test_idx

def validate_scaffold_split(train_idx, val_idx, test_idx, y_labels, task_cols):
    """
    Calculates the number of toxic compounds (1) per train, val and test split
    Parameters:
    ----------
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]
    y_labels : List[int]
    task_cols: List[str]
    ----------
    Returns
        # of toxic compounds in each splits
    num_train_pos:int 
    num_val_pos:int
    num_test_pos:int
    
    """
    for col_idx, col_name in enumerate(task_cols):
        num_train_pos = np.sum(y_labels[train_idx, col_idx]==1)
        num_val_pos = np.sum(y_labels[val_idx, col_idx]==1)
        num_test_pos = np.sum(y_labels[test_idx, col_idx]==1)
        print(f"Task {col_name:10s} | Train Pos: {num_train_pos:4d} | Val Pos: {num_val_pos:3d} | Test Pos: {num_test_pos:3d}")
    return num_train_pos, num_val_pos, num_test_pos

def inspect_model_weights(model):
    """
    Extracts L2 weight norms and gradient norms for each layer in the GNN.
    """
    weight_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            w_norm = param.data.norm(2).item()
            g_norm = param.grad.norm(2).item() if param.grad is not None else 0.0
            weight_stats[name] = {'weight_norm': w_norm, 'grad_norm': g_norm}
    return weight_stats

def log_weights_to_tensorboard(writer, model, step):
  """Logs weight norms, gradient norms, and layer histograms to TensorBoard."""
  for name, param in model.named_parameters():
    if param.requires_grad:
      tb_name = name.replace('.', '/')
      # Log scalar norms
      w_norm = param.data.norm(2).item()
      writer.add_scalar(f'Weights_L2/{tb_name}', w_norm, step)

      if param.grad is not None:
        g_norm = param.grad.norm(2).item()
        writer.add_scalar(f'Gradients_L2/{tb_name}', g_norm, step)
      writer.add_histogram(f'Histograms_Weights/{tb_name}', param.data, step)

def round_to_4(value):
    """
    Rounds the value to 4 decimal places.
    Parameters
    ----------
    value : float
        Value to round.
    Returns
    -------
    float
        Rounded value.
    """
    if isinstance(value, (float, np.floating)):
        return np.round(value, 4)
    return value
