# Our chemCPA-inspired Model

from abc import abstractmethod
from typing import List, Dict

import numpy as np
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from jeffrey.molecule_transformer import MoleculeTransformer


CELL_TYPES = ['NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells', 'B cells', 'Myeloid cells']


def one_hot_cell_type(cell_type: str):
    encoding = torch.zeros(len(CELL_TYPES), dtype=torch.float32)
    encoding[CELL_TYPES.index(cell_type)] = 1
    return encoding


def one_hot_molecules(molecule_idx_map: Dict[str, int], smiles):
    encoding = torch.zeros(len(molecule_idx_map.items()), dtype=torch.float32)
    encoding[molecule_idx_map[smiles]] = 1
    return encoding


class GeneDataset(Dataset):
    def __init__(self, data):
        self.data = data[data['control'] == False]
        self.molecule_idx_map = {}
        for i, smiles in enumerate(set(self.data['SMILES'])):
            self.molecule_idx_map[smiles] = i
        self.num_molecules = len(self.molecule_idx_map.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        control_index = row.index.get_loc('control')
        expression_row = row.iloc[control_index + 1:]
        expression_tensor = torch.tensor(expression_row.values.astype(np.float64), dtype=torch.float32)

        return {
            'cell_type': row['cell_type'],
            'cell_type_encoded': one_hot_cell_type(row['cell_type']),
            'cell_type_idx': CELL_TYPES.index(row['cell_type']),
            'sm_name': row['sm_name'],
            'sm_lincs_id': row['sm_lincs_id'],
            'SMILES': row['SMILES'],
            'SMILES_encoded': one_hot_molecules(self.molecule_idx_map, row['SMILES']),
            'expression': expression_tensor
        }

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class GeneAutoencoder(nn.Module):
    def __init__(self, num_genes, intermediate_dim, latent_dim):
        super().__init__()

        # Gene encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, intermediate_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(intermediate_dim, latent_dim)
        )
        init_weights(self.encoder)

        # Gene decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(intermediate_dim, num_genes),
        )
        init_weights(self.decoder)

    def forward(self, x):
        z = self.encoder(x)
        return z


class MoleculeEncoder(nn.Module):
    @abstractmethod
    def forward(self, smiles, latent_dim: int) -> Tensor:
        pass


class DrugPerturbationEncoder(nn.Module):
    def __init__(self, num_cell_types, molecule_encoder, latent_dim):
        super().__init__()

        self.num_cell_types = num_cell_types
        self.latent_dim = latent_dim
        self.cell_embedder = nn.Embedding(num_cell_types, latent_dim)

        self.molecule_encoder = molecule_encoder
        self.cell_scale = nn.Parameter(torch.tensor(1.0))
        self.drug_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, cell_type, smiles):
        cell_emb = self.cell_embedder(cell_type)
        drug_emb = self.molecule_encoder(smiles, self.latent_dim)

        print(f'Cell scale: {self.cell_scale}')
        print(f'Drug scale: {self.drug_scale}')

        return (self.cell_scale * cell_emb) + (self.drug_scale * drug_emb)


class AdversarialClassifiers(nn.Module):
    def __init__(self, latent_dim, num_cell_types, num_molecules):
        super().__init__()

        self.cell_classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_cell_types),
            nn.Softmax(dim=1),
        )
        init_weights(self.cell_classifier)

        self.molecule_classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_molecules),
            nn.Softmax(dim=1),
        )
        init_weights(self.molecule_classifier)

    def forward(self, z):
        cell_class = self.cell_classifier(z)
        molecule_class = self.molecule_classifier(z)
        return cell_class, molecule_class


class TransformerMoleculeEncoder(MoleculeEncoder):
    def __init__(self, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.transformer_encoder = MoleculeTransformer()
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        self.latent_dim = latent_dim
        self.smiles_cache = {}

    def forward(self, smiles: List[str], latent_dim: int) -> Tensor:
        smiles_list = []
        for smiles_str in smiles:
            if smiles_str not in self.smiles_cache:
                self.smiles_cache[smiles_str] = mol_to_graph(Chem.MolFromSmiles(smiles_str))

            smiles_list.append(graph_to_transformer_rep(self.smiles_cache[smiles_str], self.transformer_encoder))
        
        # for emb in smiles_list:
        #     print(emb.size())
        embeddings = torch.stack(smiles_list)
        embeddings = embeddings.squeeze(1)
        # print(embeddings)
        # print(embeddings.size())

        embeddings = self.batch_norm(embeddings)

        return embeddings


def smiles_to_transformer_rep(mol_smiles, tm_model):
    return graph_to_transformer_rep(mol_to_graph(Chem.MolFromSmiles(mol_smiles)), tm_model)


def data_to_graph(data):
    new_graph = AttrDict()
    new_graph.update(data.to_dict())
    return new_graph


def mol_to_graph(mol):
    graph_as_data = mol_to_pyg_graph(mol)
    graph = data_to_graph(graph_as_data)
    graph.idx = 0
    graph.y = np.array([0.0])

    return graph


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def graph_to_transformer_rep(graph, tm_model):
    global device

    max_node = 512
    multi_hop_max_dist = 5
    spatial_pos_max = 1024

    for idx, val in graph.items():
        if isinstance(val, np.ndarray):
            graph[idx] = torch.from_numpy(val)

    with torch.no_grad():
        return forward_through_graph_encoder(graph, tm_model)


def forward_through_graph_encoder(collated, transformerm_model):

    inner_states, atom_output = transformerm_model.molecule_encoder(
        collated,
        segment_labels=None,
        perturb=None,
        last_state_only=True
    )

    last_state = inner_states[0]
    molecule_embedding = last_state.permute(1, 0, 2).mean(dim=1)
    return molecule_embedding

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def mol_to_pyg_graph(mol):
    # atoms
    mol = Chem.AddHs(mol)

    bad = False

    # rdDepictor.Compute2DCoords(mol)
    try:
        if AllChem.EmbedMolecule(mol) == -1:
            bad = True
    except Exception as _:
        pass
    # AllChem.EmbedMolecule(mol)

    mol_try = Chem.Mol(mol)
    if not bad:
        try:
            AllChem.MMFFOptimizeMolecule(mol_try)
            mol = mol_try
        except Exception as _:
            pass

    mol = Chem.RemoveHs(mol)

    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        # atom_feature = [allowable_features['possible_atomic_num_list'].index(
        #     atom.GetAtomicNum())] + [allowable_features[
        #     'possible_chirality_list'].index(atom.GetChiralTag())]
        # atom_features_list.append(atom_feature)
        atom_features_list.append(atom_to_feature_vector(atom))

    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    # positions
    try:
        if not bad:
            positions = mol.GetConformer().GetPositions()
        else:
            num_atoms = mol.GetNumAtoms()
            positions = np.zeros((num_atoms, 3))
    except ValueError:
        return None
    # bonds
    # num_bond_features = 2   # bond type, bond direction
    # if len(mol.GetBonds()) > 0: # mol has bonds
    #     edges_list = []
    #     edge_features_list = []
    #     for bond in mol.GetBonds():
    #         i = bond.GetBeginAtomIdx()
    #         j = bond.GetEndAtomIdx()
    #         edge_feature = [allowable_features['possible_bonds'].index(
    #             bond.GetBondType())] + [allowable_features[
    #                                         'possible_bond_dirs'].index(
    #             bond.GetBondDir())]
    #         edges_list.append((i, j))
    #         edge_features_list.append(edge_feature)
    #         edges_list.append((j, i))
    #         edge_features_list.append(edge_feature)

    #     # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    #     edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    #     # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    #     edge_attr = torch.tensor(np.array(edge_features_list),
    #                              dtype=torch.long)
    # else:   # mol has no bonds
    #     edge_index = torch.empty((2, 0), dtype=torch.long)
    #     edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    positions = positions.astype(np.float32)

    data = Data(x=x, edge_index=torch.from_numpy(edge_index).to(torch.int64),
                edge_attr=torch.from_numpy(edge_attr).to(torch.int64))

    data.__num_nodes__ = len(x)
    data.pos = torch.from_numpy(positions)

    # for key in data.keys:
    #     attr = getattr(data, key)
    #     print(f"{key}: {attr.dtype}")

    return data
