import glob
import re

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphCPA import GeneDataset, GeneAutoencoder, TransformerMoleculeEncoder, DrugPerturbationEncoder, \
    AdversarialClassifiers, \
    CELL_TYPES


def save_models(epoch, autoencoder, molecule_encoder, perturbation_encoder,
                adv_classifiers, optimizer):
    save_dict = {'epoch': epoch,
                 'autoencoder': autoencoder.state_dict(),
                 'molecule_encoder': molecule_encoder.state_dict(),
                 'perturbation_encoder': perturbation_encoder.state_dict(),
                 'adv_classifiers': adv_classifiers.state_dict(),
                 'optimizer': optimizer.state_dict()}
    torch.save(save_dict, f'checkpoints-fixed/ckpt_epoch_{epoch}.pth')


def numeric_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', text)]


def load_models(autoencoder, molecule_encoder, perturbation_encoder, adv_classifiers, device, optimizer):
    ckpts = glob.glob('checkpoints-fixed/ckpt_epoch_*.pth')
    ckpts.sort(key=numeric_keys)
    if ckpts:
        last_ckpt = ckpts[-1]
        print(f'Loading checkpoint: {last_ckpt}')
        ckpt = torch.load(last_ckpt, map_location=device)
        epoch = ckpt['epoch'] + 1  # next epoch to start with
        autoencoder.load_state_dict(ckpt['autoencoder'])
        molecule_encoder.load_state_dict(ckpt['molecule_encoder'])
        perturbation_encoder.load_state_dict(ckpt['perturbation_encoder'])
        adv_classifiers.load_state_dict(ckpt['adv_classifiers'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f'Loaded model at epoch {epoch-1}')
    else:
        init_mol_encoder_ckpt = 'base_checkpoint/model-epoch=394.ckpt'
        #print(f'Loading checkpoint: {last_ckpt}')
        ckpt = torch.load(init_mol_encoder_ckpt, map_location=device)['state_dict']
        #print(ckpt)
        #exit()
        #missing_keys, unexpected_keys = molecule_encoder.load_state_dict(ckpt)
        #print(f'Missing: {missing_keys}')
        #print(f'Unexpected: {unexpected_keys}')
        graph_encoder_state_dict = {}
        for k, v in ckpt.items():
            if k.startswith('graph_encoder.'):
                graph_encoder_state_dict['transformer_encoder.' + k[14:]] = v
        #elif k.startswith('text_encoder.'):
        #    text_encoder_state_dict[k.removeprefix('text_encoder.')] = v
        missing_keys, unexpected_keys = molecule_encoder.load_state_dict(graph_encoder_state_dict, strict=False)
        print(f'Missing: {missing_keys}')
        print(f'Unexpected: {unexpected_keys}')
        epoch = 1 # next epoch to start with
        print(f'Loaded BASE model at epoch {epoch-1}')

    return autoencoder, molecule_encoder, perturbation_encoder, adv_classifiers, epoch, optimizer


def train(data_loader, autoencoder, perturbation_encoder, adv_classifiers, scheduler=None, start_epoch=1, epochs=100, optimizer=None):
    cross_entropy = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    autoencoder = autoencoder.to(device)
    perturbation_encoder = perturbation_encoder.to(device)
    perturbation_encoder.molecule_encoder = perturbation_encoder.molecule_encoder.to(device)
    adv_classifiers = adv_classifiers.to(device)

    for epoch in tqdm(range(start_epoch, epochs + 1)):
        for i, data in enumerate(data_loader):
            gene = data['expression'].to(device)
            cell_type = data['cell_type']
            cell_type_encoded = data['cell_type_encoded'].long().to(device)
            cell_type_indices = data['cell_type_idx'].to(device)
            SMILES = data['SMILES']
            SMILES_encoded = data['SMILES_encoded'].long().to(device)

            # Forward
            optimizer.zero_grad()
            z = autoencoder(gene)
            z_drug_attr = perturbation_encoder(cell_type_indices, SMILES)

            z_prime = z + z_drug_attr
            gene_rec = autoencoder.decoder(z_prime)
            cell_class, molecule_class = adv_classifiers(z)

            # Compute loss
            loss_rec = mse(gene_rec, gene)
            print(str(cell_class.shape) + "|" + str(cell_type_encoded.shape))
            loss_class_cell = cross_entropy(cell_class,torch.argmax(cell_type_encoded, dim=1))
            loss_class_molecules = cross_entropy(molecule_class, torch.argmax(SMILES_encoded, dim=1))
            loss = loss_rec - (loss_class_cell + loss_class_molecules)

            # Backward
            loss.backward()
            optimizer.step()

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Printing loss values
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{epochs}: Loss Rec:{loss_rec.item()} \
                    Loss Cell:{loss_class_cell.item()} \
                    Loss Molecule:{loss_class_molecules.item()}")

        # Save every 10th
        if epoch % 10 == 0:
            save_models(epoch, autoencoder, molecule_encoder, perturbation_encoder, adv_classifiers, optimizer)


if __name__ == '__main__':
    dataset = GeneDataset(pd.read_parquet('de_train.parquet'))
    data_loader = DataLoader(dataset, batch_size=8192, shuffle=True, pin_memory=True)
    print('Loaded dataset')
    autoencoder = GeneAutoencoder(18211, 512, 768)

    molecule_encoder = TransformerMoleculeEncoder(768)
    for param in molecule_encoder.parameters():
        param.requires_grad = False

    perturbation_encoder = DrugPerturbationEncoder(6, molecule_encoder, 768)
    adversarial_classifiers = AdversarialClassifiers(768, len(CELL_TYPES), dataset.num_molecules)

    param_groups = [
        {'params': autoencoder.parameters(), 'lr': 1.12e-3},
        {'params': [param for param in perturbation_encoder.parameters() if param.requires_grad], 'lr': 5.61e-4},
        {'params': adversarial_classifiers.parameters(), 'lr': 8.06e-4}
    ]
    # Initialize the optimizer with these parameter groups
    optimizer = AdamW(param_groups)

    autoencoder, molecule_encoder, perturbation_encoder, adversarial_classifiers, start_epoch, optimizer = \
        load_models(autoencoder, molecule_encoder, perturbation_encoder, adversarial_classifiers, 'cpu', optimizer)

    epochs = 1000
    train(data_loader, autoencoder, perturbation_encoder, adversarial_classifiers, None, start_epoch, epochs, optimizer)
