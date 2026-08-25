# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:47:55 2022

@author: pedro
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Linear

from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import accuracy_score , roc_auc_score , precision_score , recall_score
from sklearn.model_selection import train_test_split
import seaborn as  sns
import optuna

import shutil
import random
import warnings

df=pd.read_csv('tox21_dataset.csv')
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


def save_ckp(state, is_best, checkpoint_dir, best_model_dir, filename, best_model):
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + best_model
        shutil.copyfile(f_path, best_fpath)
        
def optimizer_to(optim, device):
    # move optimizer to device
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(checkpoint_fpath, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)

    optimizer_to(optimizer, device)
    return model, optimizer, checkpoint['epoch']


#-----------------------------------------------------------------------------------------  
embedding_size = 400
#-----------------------------------------------------------------------------------------  

CUDA_LAUNCH_BLOCKING=1

class GNN(torch.nn.Module):
    def __init__(self, layer_types, hidden_dim, dropout):
        super(GNN, self).__init__()
        self.layer_types = layer_types   # e.g. ['gcn', 'sage', 'gat']
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        in_dim = 79  # input node feature size

        for lt in layer_types:
            if lt == 'gcn':
                # This is your self.initial_conv / self.conv1 / self.conv2
                self.convs.append(GCNConv(in_dim, hidden_dim))
            elif lt == 'gat':
                # This is your self.initial_att / self.att1 / self.att2
                self.convs.append(GATConv(in_dim, hidden_dim))
            elif lt == 'gin':
                # This is your self.ginconv1 / ginconv2 / ginconv3
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                self.convs.append(GINConv(mlp, eps=0.00005, train_eps=True))
            elif lt == 'sage':
                # This is your self.initial_sage / self.sage1 / self.sage2
                self.convs.append(SAGEConv(in_dim, hidden_dim))
            # after first layer, input dim becomes hidden_dim
            in_dim = hidden_dim

        self.drop = nn.Dropout(p=dropout)
        self.out = Linear(hidden_dim * 2, 1)   # same as before

    def forward(self, x, edge_index, batch_index):
        # This loop replaces your manual sequence of hidden = layer(...)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.tanh(x)

        x = self.drop(x)

        # Same global pooling + output as before
        x = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
        x = self.out(x)
        return x

model = GNN(layer_types=['sage', 'sage', 'sage'], hidden_dim=400, dropout=0.25)
print(model)

warnings.filterwarnings("ignore")

optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#-----------------------------------------------------------------------------------------   
#batch size:
NUM_GRAPHS_PER_BATCH = 64
#Loss function weight:
pos_weight = torch.FloatTensor([6]).to(device)

criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#-----------------------------------------------------------------------------------------   
# Wrap data in a data loader
data_size = len(data_list)
# Split into train, validation, test (60/20/20)
train_idx, temp_idx = train_test_split(range(data_size), test_size=0.4, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

loader = DataLoader([data_list[i] for i in train_idx], batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False, drop_last=False)
test_loader = DataLoader([data_list[i] for i in test_idx], batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False, drop_last=False)
# loader = DataLoader(data_list[:int(data_size * 0.85)], 
#                     batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
# test_loader = DataLoader(data_list[int(data_size * 0.85):], 
#                          batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last = True)

def train(loader):
    # Enumerate over the data
    for batch in loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients
      optimizer.zero_grad() 
      # Passing the node features and the connection info
      pred = model(batch.x.float(), batch.edge_index, batch.batch)
 
      pred=torch.reshape(pred,(NUM_GRAPHS_PER_BATCH,))

      loss = criterion(pred,batch.y)
            
      loss.backward()  # Calculating the loss and gradients

      optimizer.step()   # Update using the gradients
    return loss,  optimizer

def test(loader):
    model.eval()
    y_score=[]
    y_true=[]
    for batch in loader:
      batch.to(device)  
      pred = model(batch.x.float(), batch.edge_index, batch.batch)
      pred=torch.sigmoid(pred)
      
      pred = torch.where(pred>0.5, torch.ones_like(pred), pred)
      pred = torch.where(pred<=0.5, torch.zeros_like(pred), pred)


      y_score.append(pred)
      y_true.append(batch.y)

    y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
    y_score = torch.cat(y_score, dim = 0).cpu().detach().numpy()

    y_true = [int(x) for x in y_true]
    y_score = [int(x) for x in y_score]
    accuracy_score(y_true , y_score)
    roc_list=[]
    roc_list.append(roc_auc_score(y_true, y_score))
    acc = sum(roc_list)/len(roc_list)

    precision = precision_score(y_true, y_score)
    specificity = recall_score(y_true, y_score)

    cm=confusion_matrix(y_true=y_true,y_pred=y_score)
    return acc, precision, specificity, cm

print("Starting training...")
losses = [0]
acc_list = [0]
precision_list=[0]
specificity_list=[0]
def objective(trial):
    # Suggest hyperparameters
    layer_types = []
    for i in range(3):  # fixed 3 layers
        layer_types.append(trial.suggest_categorical(f'layer_{i}', ['gcn', 'gat', 'gin', 'sage']))
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    # pos_weight is fixed based on dataset imbalance

    # Create model
    model = GNN(layer_types, hidden_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.FloatTensor([6]).to(device)  # adjust if needed
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop (early stop after 30 epochs to save time)
    best_val_auc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(50):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x.float(), batch.edge_index, batch.batch).squeeze()
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        y_val_true = []
        y_val_score = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = torch.sigmoid(model(batch.x.float(), batch.edge_index, batch.batch).squeeze())
                y_val_true.append(batch.y.cpu().numpy())
                y_val_score.append(pred.cpu().numpy())
        y_val_true = np.concatenate(y_val_true)
        y_val_score = np.concatenate(y_val_score)
        val_auc = roc_auc_score(y_val_true, y_val_score)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_auc

# --- Optuna ----
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)
# --- Save Optuna results to a text file ---
with open("optuna_results.txt", "w") as f:
    f.write("Optuna Study Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Best trial value (AUC): {study.best_trial.value:.6f}\n")
    f.write("Best hyperparameters:\n")
    for key, value in study.best_trial.params.items():
        f.write(f"  {key}: {value}\n")
    f.write("\nAll trials (sorted by AUC, descending):\n")
    f.write("=" * 60 + "\n")
    # Sort trials by value (best first)
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True
    )
    for i, trial in enumerate(sorted_trials):
        f.write(f"{i+1}. Trial #{trial.number}: AUC={trial.value:.6f} | Params={trial.params}\n")
    f.write("=" * 60 + "\n")
print(" Optuna results saved to 'optuna_results.txt'")

print("Best trial:")
best_trial = study.best_trial
print(f"  Value (AUC): {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Extract best hyperparameters
best_layer_types = [best_trial.params[f'layer_{i}'] for i in range(3)]
best_hidden = best_trial.params['hidden_dim']
best_dropout = best_trial.params['dropout']
best_lr = best_trial.params['lr']


# Final model with best hyperparameters
final_model = GNN(best_layer_types, best_hidden, best_dropout).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([6]).to(device))

# Combine train+val for final training
final_train_loader = DataLoader([data_list[i] for i in train_idx + val_idx], 
                                batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)

losses = []
acc_list = []
import time
start_time = time.time()
for epoch in range(100):  # more epochs now
    final_model.train()
    for batch in final_train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = final_model(batch.x.float(), batch.edge_index, batch.batch).squeeze()
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
    # Evaluate on test set every 10 epochs
    if epoch % 10 == 0:
        final_model.eval()
        y_test_true = []
        y_test_score = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = torch.sigmoid(final_model(batch.x.float(), batch.edge_index, batch.batch).squeeze())
                y_test_true.append(batch.y.cpu().numpy())
                y_test_score.append(pred.cpu().numpy())
        y_test_true = np.concatenate(y_test_true)
        y_test_score = np.concatenate(y_test_score)
        test_auc = roc_auc_score(y_test_true, y_test_score)
        acc_list.append(test_auc)
        elapsed_time = time.time() - start_time
        
        mins, secs = divmod(elapsed_time, 60)
        
        print(f"Epoch {epoch:03d} | Test AUC: {test_auc:.4f} | Time Elapsed: {int(mins)}m {int(secs)}s")

print(f"Final test AUC: {max(acc_list):.4f}")


# Test Epochs


# for epoch in range(100): #10_000
#     best_epoch = False
#     loss, optimizer = train(loader)
#     losses.append(loss)
#     roc_auc, precision, specificity, cm = test(test_loader)
    
#     if roc_auc >= max(acc_list):
#         best_epoch = True

#     acc_list.append(roc_auc)
#     precision_list.append(precision)
#     specificity_list.append(specificity)
    
#     checkpoint_gnn = {'epoch': epoch + 1,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict()}

   
# #----------------------------------------------------------------------------------------------
# #             Give a name to the file 
#     NAME = "sage_sage_sage"
#     checkpoint_dir = "checkpoints_tox21/" + NAME  # Give a name to the file 
#     model_dir = "checkpoints_tox21/" + NAME + "_model" # Give a name to the file 
    
#     name = "results_tox21/" + NAME + ".txt"        # Give a name to the file 
#     file = open(name,"a")
#     file.write("roc_auc: ")
#     file.write(str(roc_auc))
#     file.write("\n")
#     file.close()
    
    
#     save_ckp(checkpoint_gnn, best_epoch, checkpoint_dir, model_dir, "/checkpoints_" + NAME + "_tox21.pt", "/model_" + NAME + "_tox21.pt" )
    

#     if epoch % 20==0:
#       print(max(acc_list))
#       print(f"Epoch {epoch} | Train Loss {loss} | roc auc score {roc_auc}")
#       sns.set(style="darkgrid")
#       acc_indices = [i for i,l in enumerate(acc_list)]
#       grafico = sns.lineplot(x=acc_indices, y=acc_list)
#       plt.show()