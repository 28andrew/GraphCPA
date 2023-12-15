import sys

import pandas as pd
import torch
from tqdm import tqdm

from graphCPA import GeneAutoencoder, TransformerMoleculeEncoder, DrugPerturbationEncoder, CELL_TYPES

EPOCH_NUM = sys.argv[1] if len(sys.argv) > 1 else 500
try:
    EPOCH_NUM=int(EPOCH_NUM)
except ValueError:
    print('Invalid epoch number')
    exit(1)

print(EPOCH_NUM)

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')
    #torch.set_default_device(device)
elif torch.backends.mps.is_available():
    device = torch.device('mps')

def predict(perturb_encoder, autoencoder, cell_type, smiles):
    # Convert inputs to tensor
    cell_type_tensor = torch.tensor([cell_type]).to(device)  # If cell_type is an integer
    smiles = [smiles]  # Make sure the SMILES input is a list of strings

    # Enable evaluation mode
    perturb_encoder.eval()

    # Generate encoding for cell type and molecule
    emb = perturb_encoder(cell_type_tensor, smiles)

    # Plug the combined representation into the decoder part of the model.
    prediction = autoencoder.decoder(emb)
    prediction = prediction.cpu()
    return prediction


if __name__ == '__main__':
    #molecule_encoder = RDKitMoleculeEncoder(768)
    molecule_encoder = TransformerMoleculeEncoder(768)
    perturb_encoder = DrugPerturbationEncoder(len(CELL_TYPES), molecule_encoder, 768)
    autoencoder = GeneAutoencoder(18211, 512, 768)
    ckpt = torch.load(f'./checkpoints-fixed/ckpt_epoch_{EPOCH_NUM}.pth', map_location=device)

    molecule_encoder.load_state_dict(ckpt['molecule_encoder'])
    perturb_encoder.load_state_dict(ckpt['perturbation_encoder'])
    autoencoder.load_state_dict(ckpt['autoencoder'])
    molecule_encoder = molecule_encoder.to(device)
    # #molecule_encoder = TransformerMMoleculeEncoder(768)
    perturb_encoder = perturb_encoder.to(device)
    autoencoder = autoencoder.to(device)
    
    id_map = df = pd.read_csv('../id_map.csv')
    de_train = pd.read_parquet('../de_train.parquet')

    df = pd.read_csv('../sample_submission.csv')
    for idn in tqdm(df['id']):
        id_map_row = df[df['id'] == idn]
        cell_type = id_map['cell_type'][idn]
        sm_name = id_map['sm_name'][idn]
        smiles = list(de_train[de_train['sm_name'] == sm_name]['SMILES'])[0]
        prediction = predict(perturb_encoder, autoencoder, CELL_TYPES.index(cell_type), smiles)
        df.loc[df['id'] == idn, df.columns[1:]] = prediction.detach().numpy().tolist()

    print('Done')
    df.to_csv(f'results-fixed/inference_{EPOCH_NUM}.csv', index=False)

