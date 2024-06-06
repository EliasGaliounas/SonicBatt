# %%
from SonicBatt import utils
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell cccv')
data_path = os.path.join(study_path, 'Raw Data')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')
models_path = os.path.join(study_path, 'Classification')

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

label_column = ('cycling', 'Cell_ID')
# Replace cell_ids with the cell aliases
df[label_column] = df[label_column].replace(cell_aliases)
y = df.loc[:, label_column].to_numpy()
# Labels must start at zero!
y -= 1
# I am using the SparseCategoricalCrossEntropy function which expects labels to be ingeger
# values. If it is desirable to use one-hot encoded abels, use the CategoricalCrossEntropy instead.

with open(os.path.join(ancillary_data_path,'cells_together_split.json'), 'r') as fp:
    cells_together_split = json.load(fp)
with open(os.path.join(ancillary_data_path,'cells_separated_splits.json'), 'r') as fp:
    cells_separated_splits = json.load(fp)

# %%
representations = [
    'ffnn_1x10_sch2',
    'ffnn_1x20_sch2',
    'ffnn_1x50_sch2',
    'ffnn_1x75_sch2',
    'ffnn_1x100_sch2',
    'ffnn_2x10_sch2',
    'ffnn_2x20_sch2',
    'ffnn_2x50_sch2',
    'ffnn_2x75_sch2',
    'ffnn_2x100_sch2',
]

data_configs = {
    'A': [('peak_tofs', '8'), ('peak_heights', '8')],
    'B': ['peak_tofs'],
    'C': ['peak_tofs', 'peak_heights'],
    'D': ['acoustics'],
    'E': ['fft_magns'],
    'F': ['acoustics', 'fft_magns'],
    # 'G': 'spectrograms'
}

def config_data(data_config, Fold = None):
    if Fold == None:
        train_indices = cells_together_split['train']
        val_indices = cells_together_split['val']
        test_indices = cells_together_split['test']
    else:
        train_indices = cells_separated_splits[Fold]['train']
        val_indices = cells_separated_splits[Fold]['val']
        test_indices = cells_separated_splits[Fold]['test']        
    #
    if data_config != 'G':
        feature_columns = data_configs[data_config]
        X_train = df.loc[train_indices, feature_columns].to_numpy()
        X_val = df.loc[val_indices, feature_columns].to_numpy()
        X_test = df.loc[test_indices, feature_columns].to_numpy()
    # else:
    #     X_train = spectrograms[train_indices]
    #     X_val = spectrograms[val_indices]
    #     X_test = spectrograms[test_indices]
    #
    y_train = df.loc[train_indices, label_column].to_numpy().reshape(-1,1)
    y_val = df.loc[val_indices, label_column].to_numpy().reshape(-1,1)
    y_test = df.loc[test_indices, label_column].to_numpy().reshape(-1,1)
    #
    return (X_train, y_train, X_val, y_val, X_test, y_test)

# %%
data_config = 'A'
X_train, y_train, X_val, y_val, X_test, y_test = config_data(data_config)

representation = 'ffnn_2x10_sch2'
model_name = '{}_{}'.format(representation, data_config)
save_dir_model = os.path.join(models_path, model_name)
model = load_model(os.path.join(save_dir_model, model_name + '.h5'))

# y_pred_train = model.predict(X_train)
# y_pred_val = model.predict(X_val)
# y_pred_test = model.predict(X_test)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
y_pred_train = np.argmax(probability_model.predict(X_train), axis=1)
y_pred_val = np.argmax(probability_model.predict(X_val), axis=1)
y_pred_test = np.argmax(probability_model.predict(X_test), axis=1)

# %%
import seaborn as sns
confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=[0, 1, 2, 3, 4, 5, 6],
            yticklabels=[0, 1, 2, 3, 4, 5, 6],
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()