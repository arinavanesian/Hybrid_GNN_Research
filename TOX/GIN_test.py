# import torch
# from torch_geometric.nn import GINConv
# import pandas as pd

# class GINTest(torch.nn.Module):
#     def __init__(self, embedding_size):
#         super(GINTest, self).__init__()
#         self.mlp1 = torch.nn.Sequential(
#             torch.nn.Linear(79, embedding_size),
#             torch.nn.BatchNorm1d(embedding_size),
#             torch.nn.ReLU(),
#             torch.nn.Linear(embedding_size, embedding_size),
#             torch.nn.ReLU()
#         )
#         self.ginconv1 = GINConv(self.mlp1, eps=0.00005, train_eps=True)
#         self.out_mlp1 = torch.nn.Linear(embedding_size, embedding_size)

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# 1. Define sample test molecules (SMILES)
test_molecules = {
    "Aspirin (Aromatic Ring)": "CC(=O)Oc1ccccc1C(=O)O",
    "Nicotine (Heteroatoms in Ring)": "CN1CCC[C@H]1c2cccnc2",
    "Hexane (Acyclic / No Rings)": "CCCCCC",
    "Ethanol (Small Acyclic)": "CCO"
}
def murcko_scaffold(smiles: str)-> tuple:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"
    try:
       scaffold =MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold =""
    cannonical_smiles =Chem.MolToSmiles(mol, canonical=True)
    return (scaffold, False) if scaffold != "" else (cannonical_smiles, True)


def generic_scaffold(smiles: str)-> tuple:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"
    try:
        gen_mol = MurckoScaffold.MakeScaffoldGeneric(mol)
        gen_scaffold = Chem.MolToSmiles(gen_mol)
    except Exception:
        gen_scaffold = ""
    cannonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return (gen_scaffold,False) if gen_scaffold != "" else (cannonical_smiles,True)

import pandas as pd
df = pd.DataFrame(columns=['name', 'smiles'])
for k, v in test_molecules.items():
    df['name'] = k
    df['smiles'] = v
    df[["murcko_scaffold", "is_cannonical"]] = pd.DataFrame(
    df["smiles"].apply(murcko_scaffold).tolist(), index=df.index
)
    df[["generic_scaffold", "is_cannonical"]] = pd.DataFrame(
    df["smiles"].apply(generic_scaffold).tolist(), index=df.index
    )

print(df)