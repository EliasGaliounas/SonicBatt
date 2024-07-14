# %%
from SonicBatt import utils
import os
import pandas as pd
import numpy as np
import pickle
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell_ml')
data_path = os.path.join(study_path, 'Raw Data')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')
unsupervised_models_path = os.path.join(study_path, 'Models', 'Unsupervised')

database = pd.read_excel(os.path.join(data_path, 'database.xlsx'))
df_cell_aliases =  pd.read_excel(os.path.join(data_path, 'database.xlsx'),
                              sheet_name='cell_aliases')
cell_aliases = {}
for _, row in df_cell_aliases.iterrows():
    cell_aliases[row['cell_id']] = row['cell_alias']

parquet_filename = 'signals_peaks_fft.parquet'
parquet_filepath = os.path.join(ancillary_data_path, parquet_filename)
df = pd.read_parquet(parquet_filepath)
# Get rid of the invariable parts of the acoustic signals
df = df.drop(columns = [('acoustics', str(i)) for i in range(758)])
spectrograms = np.load(os.path.join(ancillary_data_path, 'spectrograms.npy'))

# Important: Shuffle the dataset in the same way it was shuffled to produce
# the cells_together_split in scipt '0_initial_data_processing'
indices_all = df.index.to_numpy().copy() # !!! The following alternative didn't work --> #df.copy(deep=True).index.to_numpy()
np.random.seed(42)
np.random.shuffle(indices_all)
df = df.loc[indices_all].reset_index(drop=True)
spectrograms = spectrograms[indices_all]

flattened_spectrograms = spectrograms.reshape(
    spectrograms.shape[0],
    spectrograms.shape[1] * spectrograms.shape[2])

with open(os.path.join(ancillary_data_path,'cells_together_split.json'), 'r') as fp:
    cells_together_split = json.load(fp)

# %%
data_configs = {
    'B': ['peak_tofs'],
    'C': ['peak_tofs', 'peak_heights'],
    'D': ['acoustics'],
    'E': ['fft_magns'],
    'F': ['acoustics', 'fft_magns'],
    'G': 'spectrograms'
}

# %%
# PCA and TSNE training
train_indices = cells_together_split['train']
test_indices = cells_together_split['test']

def train_model(model, model_name, data_config):
    if data_config != 'G':
        feature_columns = data_configs[data_config]
        X_train = df.loc[train_indices, feature_columns].to_numpy()
        X_test = df.loc[test_indices, feature_columns].to_numpy()
    else:
        X_train = flattened_spectrograms[train_indices]
        X_test = flattened_spectrograms[test_indices]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    model.fit(X_train)
    X_test = scaler.transform(X_test)
    projected = model.fit_transform(X_test)
    save_dir = os.path.join(unsupervised_models_path, model_name) + '.pkl'
    with open(save_dir, 'wb') as fa:
        pickle.dump(projected, fa)

print('Fitting PCA models')
for data_config in data_configs.keys():
    print('Working on data config {}'.format(data_config))
    model_name = 'PCA_{}'.format(data_config)
    model = PCA(n_components=2)
    train_model(model, model_name, data_config)

print('Fitting TSNE models')
for data_config in data_configs.keys():
    print('Working on data config {}'.format(data_config))
    model_name = 'TSNE_{}'.format(data_config)
    model = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
    train_model(model, model_name, data_config)

# %%