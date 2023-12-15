import pandas as pd
import torch
from tqdm import tqdm

if __name__ == '__main__':
    id_map = df = pd.read_csv('id_map.csv')
    de_train = pd.read_parquet('de_train.parquet')

    df = pd.read_csv('sample_submission.csv')
    for idn in tqdm(df['id']):
        id_map_row = df[df['id'] == idn]
        cell_type = id_map['cell_type'][idn]
        sm_name = id_map['sm_name'][idn]
        smiles = list(de_train[de_train['sm_name'] == sm_name]['SMILES'])[0]
        random_tensor = torch.randn(18211)
        df.loc[df['id'] == idn, df.columns[1:]] = random_tensor.detach().numpy().tolist()

    print('Done')
    df.to_csv('inference_random.csv', index=False)
