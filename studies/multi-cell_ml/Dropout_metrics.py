# %%
from SonicBatt import utils
import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell_ml')
data_path = os.path.join(study_path, 'Raw Data')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')
models_path = os.path.join(study_path, 'Regression')

database = pd.read_excel(os.path.join(data_path, 'database.xlsx'))
selected_cells = database.loc[:,'cell_id'].to_list()
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

label_column = ('cycling', 'V(V)')

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
    'ffnn_3x10_sch2',
    'ffnn_3x20_sch2',
    'ffnn_3x50_sch2',
    'ffnn_3x75_sch2',
    'ffnn_3x100_sch2',

    'cnn_filters_8_16_dense_1x50_sch2',
    'cnn_filters_16_32_dense_1x50_sch2',
    'cnn_filters_32_64_dense_1x50_sch2',
    'cnn_filters_8_16_dense_1x100_sch2',
    'cnn_filters_16_32_dense_1x100_sch2',
    'cnn_filters_32_64_dense_1x100_sch2',

    'cnn_filters_8_16_dense_2x50_sch2',
    'cnn_filters_16_32_dense_2x50_sch2',
    'cnn_filters_32_64_dense_2x50_sch2',
    'cnn_filters_8_16_dense_2x100_sch2',
    'cnn_filters_16_32_dense_2x100_sch2',
    'cnn_filters_32_64_dense_2x100_sch2',

    'cnn_filters_8_16_dense_3x50_sch2',
    'cnn_filters_16_32_dense_3x50_sch2',
    'cnn_filters_32_64_dense_3x50_sch2',
    'cnn_filters_8_16_dense_3x100_sch2',
    'cnn_filters_16_32_dense_3x100_sch2',
    'cnn_filters_32_64_dense_3x100_sch2'
]

data_configs = {
    'A': [('peak_tofs', '8'), ('peak_heights', '8')],
    'B': ['peak_tofs'],
    'C': ['peak_tofs', 'peak_heights'],
    'D': ['acoustics'],
    'E': ['fft_magns'],
    'F': ['acoustics', 'fft_magns'],
    'G': 'spectrograms'
}

markers = {
    'A': 'd',
    'B': 'P',
    'C': '<',
    'D': 'v',
    'E': 'o',
    'F': 's',
    'G': 'X'
}

cols = {
    'A': 'tab:purple',
    'B': 'tab:cyan',
    'C': 'tab:red',
    'D': 'tab:green',
    'E': 'tab:blue',
    'F': 'tab:orange',
    'G': 'k'
}

deep_representations = [
    'ffnn_1x100_sch2',
    'ffnn_2x100_sch2',
    'ffnn_3x100_sch2',
    'ffnn_4x100_sch2',
    'ffnn_5x100_sch2',
    'ffnn_6x100_sch2',
    'ffnn_7x100_sch2',
    'ffnn_8x100_sch2',
    'ffnn_9x100_sch2',
    'ffnn_10x100_sch2',
    'ffnn_11x100_sch2',
    'ffnn_12x100_sch2',
    'ffnn_13x100_sch2',
    'ffnn_14x100_sch2',
    'ffnn_15x100_sch2',
    'ffnn_16x100_sch2',
    'ffnn_17x100_sch2',
    'ffnn_18x100_sch2',
    'ffnn_19x100_sch2',
    'ffnn_20x100_sch2',
]

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
    else:
        X_train = spectrograms[train_indices]
        X_val = spectrograms[val_indices]
        X_test = spectrograms[test_indices]
    #
    y_train = df.loc[train_indices, label_column].to_numpy().reshape(-1,1)
    y_val = df.loc[val_indices, label_column].to_numpy().reshape(-1,1)
    y_test = df.loc[test_indices, label_column].to_numpy().reshape(-1,1)
    #
    return (X_train, y_train, X_val, y_val, X_test, y_test)

# %%
# MAE matrices
cell_config = 'cells_separated_dropout'

path_mae_all_folded_train = os.path.join(models_path, cell_config,
     'mae_all_folded_train.csv')
path_mae_all_folded_val = os.path.join(models_path, cell_config,
     'mae_all_folded_val.csv')
path_mae_all_folded_test = os.path.join(models_path, cell_config,
     'mae_all_folded_test.csv')

if (not os.path.exists(path_mae_all_folded_train)) & \
    (not os.path.exists(path_mae_all_folded_val)) & \
        (not os.path.exists(path_mae_all_folded_test)):
    first_level = list(data_configs.keys())
    second_level = ['F{}'.format(i) for i in range(1,8)]
    multi_index = pd.MultiIndex.from_product([first_level, second_level])
    mae_all_folded_train = pd.DataFrame(index=representations, columns=multi_index)
    mae_all_folded_test = pd.DataFrame(index=representations, columns=multi_index)
    mae_all_folded_val = pd.DataFrame(index=representations, columns=multi_index)
else:
    mae_all_folded_train = pd.read_csv(os.path.join(models_path, cell_config,
        'mae_all_folded_train.csv'), header=[0, 1], index_col=0)
    mae_all_folded_val = pd.read_csv(os.path.join(models_path, cell_config,
        'mae_all_folded_val.csv'), header=[0, 1], index_col=0)
    mae_all_folded_test = pd.read_csv(os.path.join(models_path, cell_config,
        'mae_all_folded_test.csv'), header=[0, 1], index_col=0)

# %%
made_changes_flag = False
for data_config in ['F']: #data_configs.keys() | ['C'] :
    print('Working on data config: {}'.format(data_config))
    for i, fold in enumerate(cells_separated_splits.keys()): # 
        X_train, y_train, X_val, y_val, X_test, y_test = config_data(data_config, fold)
        for representation in representations: #representations: | ['SVM']
            model_name = '{}_{}_{}'.format(representation, data_config, fold)
            model_dir = os.path.join(models_path, cell_config, model_name)
            if os.path.exists(model_dir) & \
                np.isnan(mae_all_folded_train.loc[representation, (data_config, fold)]):
                #
                made_changes_flag = True
                print('Working on: {}'.format(model_name))
                if (representation != 'LinReg') & (representation != 'SVM'):
                    model = load_model(os.path.join(model_dir, model_name + '.h5'))
                    # # Train
                    # pred_train = model.predict(X_train)
                    # mae_train = np.average(np.abs(y_train-pred_train))
                    # mae_all_folded_train.loc[
                    #     representation, (data_config, fold)] = round(mae_train,8)
                    # # Val
                    # pred_val = model.predict(X_val)
                    # mae_val = np.average(np.abs(y_val-pred_val))
                    # mae_all_folded_val.loc[
                    #     representation, (data_config, fold)] = round(mae_val,8)
                    # Test
                    pred_test = model.predict(X_test)
                    mae_test = np.average(np.abs(y_test-pred_test))
                    mae_all_folded_test.loc[
                        representation, (data_config, fold)] = round(mae_test,8)
                else:
                    with open(os.path.join(model_dir, model_name + '.sav'), 'rb') as fa:
                        model = pickle.load(fa)
                    with open(os.path.join(model_dir, 'scaler.sav'), 'rb') as fa:
                        scaler = pickle.load(fa)
                    # Train
                    # print('Inference on training data')
                    # pred_train = model.predict(scaler.transform(X_train))
                    # mae_train = np.average(np.abs(y_train-pred_train.reshape(-1,1)))
                    # mae_all_folded_train.loc[
                    #     representation, (data_config, fold)] = round(mae_train,8)
                    # # Val
                    # print('Inference on validation data')
                    # pred_val = model.predict(scaler.transform(X_val))
                    # mae_val = np.average(np.abs(y_val-pred_val.reshape(-1,1)))
                    # mae_all_folded_val.loc[
                    #     representation, (data_config, fold)] = round(mae_val,8)
                    # Test
                    print('Inference on test data')
                    pred_test = model.predict(scaler.transform(X_test))
                    mae_test = np.average(np.abs(y_test-pred_test.reshape(-1,1)))
                    mae_all_folded_test.loc[
                        representation, (data_config, fold)] = round(mae_test,8)

        del (X_train, y_train, X_val, y_val, X_test, y_test)

if made_changes_flag:
    print('Saving tables of mae')
    mae_all_folded_train.to_csv(path_mae_all_folded_train)
    mae_all_folded_val.to_csv(path_mae_all_folded_val)
    mae_all_folded_test.to_csv(path_mae_all_folded_test)