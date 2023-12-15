# Our chemCPA-inspired Model

from abc import abstractmethod
from typing import List, Dict

import numpy as np
import torch
from descriptastorus.descriptors import MakeGenerator
from torch import nn, Tensor
from torch.nn import Linear
from torch.utils.data import Dataset

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
    def forward(self, smiles: List[str], latent_dim: int) -> Tensor:
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


class RDKitMoleculeEncoder(MoleculeEncoder):
    def __init__(self, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = Linear(200, latent_dim)
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        init_weights(self.linear)
        self.smiles_cache = {}

    def forward(self, smiles: List[str], latent_dim: int) -> Tensor:
        fingerprints = []
        for smiles_str in smiles:
            if smiles_str not in self.smiles_cache:
                self.smiles_cache[smiles_str] = smiles_to_fingerprint(smiles_str).float().to(next(self.parameters()).device)
            fingerprints.append(self.smiles_cache[smiles_str])
        embeddings = torch.stack(fingerprints)
        linear_output = self.linear(embeddings)
        linear_output = self.batch_norm(linear_output)
        return linear_output


generator = MakeGenerator(["RDKit2D"])


def smiles_to_fingerprint(smiles):
    descriptor = generator.process(smiles)
    fingerprint = descriptor[1:]
    fingerprint_np = np.array(fingerprint)
    fingerprint_tensor = torch.from_numpy(fingerprint_np)

    return fingerprint_tensor