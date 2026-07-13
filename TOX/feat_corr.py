import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Linear

from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import accuracy_score , roc_auc_score , precision_score , recall_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
import seaborn as  sns

from random import random
from utils import *

df=pd.read_csv('/home/chemi_t/Documents/sci_fi/Programming/AUA/classes/Thesis/code/Hybrid_GNN/HybridGNN/TOX/tox21_dataset.csv')
#for i in df.columns:
    #print(i, df[i].isna().sum(), str(df[i].sum()/len(df[i])))

#df=pd.read_csv('tox21_dataset_task.csv')
df_task = pd.DataFrame()
df_task['smiles']=df['smiles']
df_task['SR-ARE']=df['SR-ARE']

df = df_task.dropna()

df['SR-ARE'].value_counts() #
df['SR-ARE'].value_counts().plot.pie(autopct='%.2f')

print('#of compound: ',len(df['SR-ARE']))

X = list(df['smiles'])
y = list(df['SR-ARE'])



def one_hot_encoding(x, permitted_list):
    """
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
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

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs 
    
    """
    
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        
        # SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
    return data_list


data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(X, y)

random.shuffle(data_list)
data = DataLoader(dataset = data_list, batch_size = 64)

#  Calculating Pearson correlation with the target variable for each feature
mol_features = []
targets = []
for data in data_list:
    mol_feat = data.x.mean(dim=0).numpy()  
    mol_features.append(mol_feat)
    targets.append(data.y.item())

mol_features = np.array(mol_features)  
targets = np.array(targets)

corr_with_target = []
for i in range(mol_features.shape[1]):
    corr, _ = pearsonr(mol_features[:, i], targets)
    corr_with_target.append(abs(corr))

# pairwise correlation between features (to drop highly correlated ones)
corr_matrix = np.corrcoef(mol_features.T)  # (79,79)
# Plot heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Pairwise Correlation of Node Features (Averaged per Molecule)')
plt.show()

# 4. Identify pairs with correlation > 0.9 (for example)
high_corr_pairs = []
for i in range(79):
    for j in range(i+1, 79):
        if abs(corr_matrix[i, j]) > 0.9:
            high_corr_pairs.append((i, j, corr_matrix[i, j]))
print(f"Number of highly correlated feature pairs (|r|>0.9): {len(high_corr_pairs)}")

# Feature importance using mutual information (optional)
mi = mutual_info_classif(mol_features, targets, random_state=42)
important_features = np.where(mi > np.percentile(mi, 75))[0] 
print(f"Features with high mutual information (top 25%): {important_features}")

# Based on correlation and MI, you can decide which features to drop.
# For example, drop features that are highly correlated with another and have lower MI.
# Here we just print suggestions.
# We'll create a set of features to keep (initially all)
features_to_keep = set(range(79))
# For each highly correlated pair, drop the one with lower MI
for i, j, _ in high_corr_pairs:
    if mi[i] < mi[j]:
        features_to_keep.discard(i)
    else:
        features_to_keep.discard(j)

print(f"Suggested features to keep (after dropping redundant ones): {sorted(features_to_keep)}")
print(f"Number of features to keep: {len(features_to_keep)}")