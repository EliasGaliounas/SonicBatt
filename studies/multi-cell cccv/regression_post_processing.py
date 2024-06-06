# %%
from SonicBatt import utils
import os
import json
import pickle
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

model_params = pd.read_excel(
    os.path.join(models_path, 'Model_parameters.xlsx'), index_col=0)

# %%
representations = [
    'LinReg',
    # 'SVM',
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
# Finding in the number of parameters for neural network models
# cell_config = 'cells_together'
# for data_config in ['F']: #data_configs.keys():
#     for representation in representations:
#         model_name = '{}_{}'.format(representation, data_config)
#         model_dir = os.path.join(models_path, cell_config, model_name)
        
#         if os.path.exists(model_dir):
#             model = load_model(os.path.join(model_dir, model_name + '.h5'))
#             print(model.name)
#             print(model.summary())
#             print('---------------------------')
#             print('')

# %%
# MAE matrices
cell_config = 'cells_together'

path_mae_all_train = os.path.join(models_path, cell_config,
     'mae_all_train.csv')
path_mae_all_val = os.path.join(models_path, cell_config,
     'mae_all_val.csv')
path_mae_all_test = os.path.join(models_path, cell_config,
     'mae_all_test.csv')

if (not os.path.exists(path_mae_all_train)) & \
    (not os.path.exists(path_mae_all_val)) & \
        (not os.path.exists(path_mae_all_test)):
    mae_all_train = pd.DataFrame(
        index = representations, columns = data_configs.keys())
    mae_all_val = pd.DataFrame(
        index = representations, columns = data_configs.keys())
    mae_all_test = pd.DataFrame(
        index = representations, columns = data_configs.keys())
else:
    mae_all_train = pd.read_csv(path_mae_all_train, index_col=0)
    mae_all_val = pd.read_csv(path_mae_all_test, index_col=0)
    mae_all_test = pd.read_csv(path_mae_all_test, index_col=0)

# %%
made_changes_flag = False
for data_config in ['E']: #data_configs.keys(): #: ['F']
    print('Working on data_config: {}'.format(data_config))
    X_train, y_train, X_val, y_val, X_test, y_test = config_data(data_config)
    for representation in representations:
        model_name = '{}_{}'.format(representation, data_config)
        model_dir = os.path.join(models_path, cell_config, model_name)
        if (os.path.exists(model_dir) &
            np.isnan(mae_all_train.loc[representation, data_config])):
            # 
            made_changes_flag = True
            print('Working on: {}'.format(model_name))
            if (representation != 'LinReg') & (representation != 'SVM'):
                model = load_model(os.path.join(model_dir, model_name + '.h5'))
                # Train
                pred_train = model.predict(X_train)
                mae_train = np.average(np.abs(y_train-pred_train))
                mae_all_train.loc[representation, data_config] = round(mae_train,8)
                # Val
                pred_val = model.predict(X_val)
                mae_val = np.average(np.abs(y_val-pred_val))
                mae_all_val.loc[representation, data_config] = round(mae_val,8)
                # Test
                pred_test = model.predict(X_test)
                mae_test = np.average(np.abs(y_test-pred_test))
                mae_all_test.loc[representation, data_config] = round(mae_test,8)
            else:
                with open(os.path.join(model_dir, model_name + '.sav'), 'rb') as fa:
                    model = pickle.load(fa)
                with open(os.path.join(model_dir, 'scaler.sav'), 'rb') as fa:
                    scaler = pickle.load(fa)
                # Train
                pred_train = model.predict(scaler.transform(X_train))
                mae_train = np.average(np.abs(y_train-pred_train.reshape(-1,1)))
                mae_all_train.loc[representation, data_config] = round(mae_train,8)
                # Val
                pred_val = model.predict(scaler.transform(X_val))
                mae_val = np.average(np.abs(y_val-pred_val.reshape(-1,1)))
                mae_all_val.loc[representation, data_config] = round(mae_val,8)
                # Test
                pred_test = model.predict(scaler.transform(X_test))
                mae_test = np.average(np.abs(y_test-pred_test.reshape(-1,1)))
                mae_all_test.loc[representation, data_config] = round(mae_test,8)

if made_changes_flag:
    print('Saving tables of mae')
    mae_all_train.to_csv(path_mae_all_train)
    mae_all_val.to_csv(path_mae_all_val)
    mae_all_test.to_csv(path_mae_all_test)

# %%
s=50
f, axs = plt.subplots(1,1,figsize=(4.5,3*2), constrained_layout=True)
f.patch.set_facecolor('white')
for data_config in data_configs.keys():
    for i, representation in enumerate(representations): #representations): #[:-1]
        if ('sch2' in representation):
            repr_short = representation[:-5]
            n_params = model_params.loc[repr_short, data_config]
        else:
            n_params = model_params.loc[representation, data_config]
        mae_test = mae_all_test.loc[representation, data_config]
        if i == 0:
            label = data_config
        else:
            label = None
        if '_2x' in representation:
            facecolors='none'
            hatch =  None # None | '///////'
        elif '_3x' in representation:
            facecolors='none'
            hatch = '||||||||'
        else:
            facecolors=cols[data_config]
            hatch = None
        axs.scatter(n_params, mae_test*1000, label=label, marker=markers[data_config],
                    facecolors=facecolors, edgecolors=cols[data_config],
                    hatch=hatch, s=s)

axs.set_yscale('log')
# axs.set_ylim(1E1, 1E2)
axs.set_xscale('log')

# Draw a horizontal line for a hypothetical estimator that aways gives y_pred = 3.89
# x_min, x_max = axs.get_xlim()
# axs.plot([x_min, x_max], [0.2554*1000, 0.2554*1000], color='k')

# Draw Line for the Linear Regression
n_params = model_params.loc['LinReg', :].to_numpy()
mae_test = mae_all_test.loc['LinReg', :].to_numpy()*1000
sorted_indices = np.argsort(n_params)
axs.plot(n_params[sorted_indices], mae_test[sorted_indices], color='k', linestyle = (0, (5, 1)),
         label = 'Lin. Reg.')

# Draw Line for SVM
n_params = model_params.loc['SVM', :].to_numpy()
mae_test = mae_all_test.loc['SVM', :].to_numpy()*1000
sorted_indices = np.argsort(n_params)
axs.plot(n_params[sorted_indices], mae_test[sorted_indices], color='k', linestyle = ':',
         label = 'SVM')
axs.set_yticks([15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200])
axs.get_yaxis().set_major_formatter(plt.ScalarFormatter())



axs.set_xlabel('Number of model parameters')
axs.set_ylabel('Mean absolute error\n(mV)', horizontalalignment='center')
f.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'Data config.')

# %%
# Diagonal plots
cell_config = 'cells_together'
data_config = 'G'
representation = 'cnn_filters_8_16_dense_3x100_sch2'

feature_columns = data_configs[data_config]

X_train, y_train, X_val, y_val, X_test, y_test = config_data(data_config)

model_name = '{}_{}'.format(representation, data_config)
save_dir_model = os.path.join(models_path, cell_config, model_name)
model = load_model(os.path.join(save_dir_model, model_name + '.h5'))
save_dir_history = os.path.join(save_dir_model, 'training_history.json')
with open(save_dir_history, 'r') as fp:
    history = json.load(fp)

pred_train = model.predict(X_train)
pred_val = model.predict(X_val)
pred_test = model.predict(X_test)

f, axs = plt.subplots(1,3, figsize=(9,3), constrained_layout=True, sharex=True, sharey=True)
for i in range(3):
    axs[i].plot([2.75, 4.2],[2.75, 4.2], color='k')
axs[0].set_xticks([2.7, 4.2])
axs[0].set_xlim(2.7, 4.25)
axs[0].set_yticks([2.7, 4.2])
axs[0].set_ylim(2.7, 4.25)
axs[0].scatter(y_train, pred_train, color='tab:blue', s=1) 
axs[1].scatter(y_val, pred_val, color='tab:orange', s=1)
axs[2].scatter(y_test, pred_test, color='tab:red', s=1)
axs[0].set_title('Training set\nMAE = {:.1f} mV'.format(
    np.average(np.abs(y_train-pred_train))*1000))
axs[1].set_title('Validation set\nMAE = {:.1f} mV'.format(
    np.average(np.abs(y_val-pred_val))*1000))
axs[2].set_title('Test set\nMAE = {:.1f} mV'.format(
    np.average(np.abs(y_test-pred_test))*1000))
f.supxlabel('V(V)\nTrue value')
f.supylabel('V(V)\nPredicted  value', ha='center')

# %%
train_indices = cells_together_split['train']
val_indices = cells_together_split['val']
test_indices = cells_together_split['test']

df_cycling = df['cycling']
df_cycling.loc[:, 'pred_train'] = None
df_cycling.loc[:, 'pred_val'] = None
df_cycling.loc[:, 'pred_test'] = None

df_cycling.loc[train_indices, 'pred_train'] = pred_train
df_cycling.loc[val_indices, 'pred_val'] = pred_val
df_cycling.loc[test_indices, 'pred_test'] = pred_test

# %%
# Cycling plots per cell
s=0.5
split_to_plot = 'test'

for cell_id in selected_cells:
    f, axs = plt.subplots(2, 1, figsize=(8,4), dpi=300,
        height_ratios = [1,4], sharex=True,
        constrained_layout=True)
    f.patch.set_facecolor('white')
    filter_cell = df_cycling['Cell_ID'] == cell_id
    filter_train = df_cycling['pred_train'] != None
    filter_val = df_cycling['pred_val'] != None
    filter_test = df_cycling['pred_test'] != None
    #
    split_filters = {
        'train': filter_train,
        'val': filter_val,
        'test':filter_test
    }
    #
    unique_cycles = df_cycling.loc[
        df_cycling['Cycle'] != 999, 'Cycle'].unique()
    for cycle in unique_cycles:
        filter_cycle = df_cycling['Cycle']==cycle
        temp_df = df_cycling.loc[filter_cell & split_filters[split_to_plot] & filter_cycle]

        time_h = temp_df['Time(s)'].to_numpy()/3600
        axs[0].plot(time_h, temp_df['C-Rate'], c='k')
        # axs[1].scatter(time_h, temp_df['V(V)'], c='tab:blue', s=s)
        axs[1].scatter(time_h, temp_df['pred_{}'.format(split_to_plot)], s=15,
                       marker='o',facecolors='none', edgecolors='tab:orange')
        # axs[1].plot(time_h, temp_df['pred_{}'.format(split_to_plot)], c='tab:orange')
    axs[0].set_xlim(20,50)
    f.suptitle('Cell {}'.format(cell_aliases[cell_id]))

# %%



# -------------------------------------------------------------
# Cells separated
# -------------------------------------------------------------




# %%
cell_config = 'cells_separated'
data_config = 'C'
representation = 'ffnn_1x100_sch2'

f, axs = plt.subplots(7, 3, figsize=(6,8), constrained_layout=True, sharex=True, sharey=True)
mae_train = []
mae_val = []
mae_test = []
for i, fold in enumerate(cells_separated_splits.keys()):
    X_train, y_train, X_val, y_val, X_test, y_test = config_data(data_config, fold)

    model_name = '{}_{}_{}'.format(representation, data_config, fold)
    save_dir_model = os.path.join(models_path, cell_config, model_name)
    model = load_model(os.path.join(save_dir_model, model_name + '.h5'))

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    error_train = np.average(np.abs(y_train-pred_train))
    error_val = np.average(np.abs(y_val-pred_val))
    error_test = np.average(np.abs(y_test-pred_test))

    mae_train.append(error_train)
    mae_val.append(error_val)
    mae_test.append(error_test)

    for j in range(3):
        axs[i,j].plot()
        axs[i,j].plot([2.75, 4.2],[2.75, 4.2], color='k')
    axs[0,0].set_xticks([2.7, 4.2])
    axs[0,0].set_xlim(2.7, 4.25)
    axs[0,0].set_yticks([2.7, 4.2])
    axs[0,0].set_ylim(2.7, 4.25)

    axs[i,0].scatter(y_train,pred_train, color='tab:blue', s=1) 
    axs[i,1].scatter(y_val,pred_val, color='tab:orange', s=1)
    axs[i,2].scatter(y_test,pred_test, color='tab:red', s=1)

    axs[i,0].set_ylabel('Fold {}'.format(fold[1:]))

axs[0,0].set_title('Training set\nMAE = {:.4f}'.format(np.average(mae_train)))
axs[0,1].set_title('Validation set\nMAE = {:.4f}'.format(np.average(mae_val)))
axs[0,2].set_title('Test set\nMAE = {:.4f}'.format(np.average(mae_test)))

f.supxlabel('V(V)\nTrue value')
f.supylabel('V(V)\nPredicted  value', ha='center')

# %%
# MAE matrices
cell_config = 'cells_separated'

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

# ['F1', 'F2', 'F3', 'F4']
# ['F5', 'F6', 'F7']
made_changes_flag = False
for data_config in ['G'] :#data_configs.keys():
    print('Working on data config: {}'.format(data_config))
    for i, fold in enumerate(['F7']):# enumerate(cells_separated_splits.keys()): # 
        X_train, y_train, X_val, y_val, X_test, y_test = config_data(data_config, fold)
        for representation in representations:
            model_name = '{}_{}_{}'.format(representation, data_config, fold)
            model_dir = os.path.join(models_path, cell_config, model_name)
            if os.path.exists(model_dir) & \
                np.isnan(mae_all_folded_train.loc[representation, (data_config, fold)]):
                #
                made_changes_flag = True
                print('Working on: {}'.format(model_name))
                if (representation != 'LinReg') & (representation != 'SVM'):
                    model = load_model(os.path.join(model_dir, model_name + '.h5'))
                    # Train
                    pred_train = model.predict(X_train)
                    mae_train = np.average(np.abs(y_train-pred_train))
                    mae_all_folded_train.loc[
                        representation, (data_config, fold)] = round(mae_train,8)
                    # Val
                    pred_val = model.predict(X_val)
                    mae_val = np.average(np.abs(y_val-pred_val))
                    mae_all_folded_val.loc[
                        representation, (data_config, fold)] = round(mae_val,8)
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
                    pred_train = model.predict(scaler.transform(X_train))
                    mae_train = np.average(np.abs(y_train-pred_train.reshape(-1,1)))
                    mae_all_folded_train.loc[
                        representation, (data_config, fold)] = round(mae_train,8)
                    # Val
                    pred_val = model.predict(scaler.transform(X_val))
                    mae_val = np.average(np.abs(y_val-pred_val.reshape(-1,1)))
                    mae_all_folded_val.loc[
                        representation, (data_config, fold)] = round(mae_val,8)
                    # Test
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

# %%
f, axs = plt.subplots(1,1,figsize=(4.5,3), constrained_layout=True)
for data_config in data_configs.keys():
    for i, representation in enumerate(representations):
        if 'sch2' in representation:
            repr_short = representation[:-5]
            n_params = model_params.loc[repr_short, data_config]
        else:
            n_params = model_params.loc[representation, data_config]
        for j, fold in enumerate(cells_separated_splits.keys()):
            # ------------------------------------------------
            # if qualifies_by_epoch.loc[representation, (data_config, fold)]:
                mae_test = mae_all_folded_test.loc[representation, (data_config, fold)]
                if (i == 2) & (j == 0):
                    label = data_config
                else:
                    label = None
                if '_2x' in representation:
                    facecolors='none'
                    hatch =  None # None | '///////'
                else:
                    facecolors=cols[data_config]
                    hatch = None
                axs.scatter(n_params, mae_test*1000, label=label, marker=markers[data_config],
                            facecolors=facecolors, edgecolors=cols[data_config],
                            hatch=hatch, s=12)
                # ------------------------------------------------

                # if not mae_all_folded_test.loc[representation, data_config].isna().all():
                #     mae_test_mean = mae_all_folded_test.loc[representation, data_config].mean()
                #     mae_test_max = mae_all_folded_test.loc[representation, data_config].max()
                #     mae_test_min = mae_all_folded_test.loc[representation, data_config].min()
                #     # if (i == 0) & (j == 0):
                #     #     label = data_config
                #     # else:
                #     #     label = None
                #     # axs.scatter(n_params, mae_test_mean*1000, label=label, marker=markers[data_config],
                #     #             facecolors=facecolors, edgecolors=cols[data_config],
                #     #             hatch=hatch, s=3)
                #     # Calculate error bar lengths
                #     label = None
                #     yerr = [
                #         [mae_test_mean*1000 - mae_test_min*1000],
                #         [mae_test_max*1000 - mae_test_mean*1000]]
                #     axs.errorbar(n_params, mae_test_mean*1000, yerr=yerr, label=label, marker=None,
                #                 markerfacecolor =facecolors, markeredgecolor=cols[data_config],
                #                 hatch=hatch, ecolor=cols[data_config], elinewidth=0.5, capsize=2, capthick=0.5)

# Draw Line for the Linear Regression
n_params = model_params.loc['LinReg', :].to_numpy()
for j, fold in enumerate(cells_separated_splits.keys()):
    mae_test = []
    n_params = []
    for data_config in data_configs.keys():
        mae_test.append(mae_all_folded_test.loc['LinReg', (data_config, fold)]*1000)
        n_params.append(model_params.loc['LinReg', data_config])
    n_params = np.array(n_params)
    mae_test = np.array(mae_test)
    sorted_indices = np.argsort(n_params)
    if j == 0:
        label = 'Lin. Reg.'
    else:
        label = None
    axs.plot(n_params[sorted_indices], mae_test[sorted_indices], color='k', linestyle = (0, (5, 1)),
            label = label)
    if j == 6:
        break

axs.set_yscale('log')
axs.set_ylim(1E1, 1E6)
axs.set_xscale('log')

f.supxlabel('Number of model parameters x 1000')
f.supylabel('Mean absolute error\n(mV)', horizontalalignment='center')
f.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'Data config.')

# %%
# Get number of training epochs for cell_config = 'cells_separated'
cell_config = 'cells_separated'
first_level = list(data_configs.keys())
second_level = ['F{}'.format(i) for i in range(1,8)]
multi_index = pd.MultiIndex.from_product([first_level, second_level])
epochs_all_folded = pd.DataFrame(index=representations, columns=multi_index)
for data_config in data_configs.keys():
    print('Working on data config: {}'.format(data_config))
    for i, fold in enumerate(cells_separated_splits.keys()):
        for representation in representations[2:]:
            model_name = '{}_{}_{}'.format(representation, data_config, fold)
            model_dir = os.path.join(models_path, cell_config, model_name)
            if os.path.exists(model_dir):
                with open(os.path.join(model_dir,'training_history.json'), 'r') as fp:
                    training_history = json.load(fp)
                    n_epochs = len(training_history['loss'])
                    epochs_all_folded.loc[
                        representation, (data_config, fold)] = n_epochs
np_epochs_all_folded = epochs_all_folded.to_numpy()
qualifies_by_epoch = epochs_all_folded >= 550


# %%
# Get number of training epochs for cell_config = 'cells_together'
cell_config = 'cells_together'
epochs_all = pd.DataFrame(index = representations, columns = data_configs.keys())
for data_config in data_configs.keys():
    for representation in representations:
        model_name = '{}_{}'.format(representation, data_config)
        model_dir = os.path.join(models_path, cell_config, model_name)        
        if os.path.exists(model_dir):
            with open(os.path.join(model_dir,'training_history.json'), 'r') as fp:
                training_history = json.load(fp)
                n_epochs = len(training_history['loss'])
                epochs_all.loc[representation, data_config] = n_epochs
np_epochs_all = epochs_all.to_numpy()

# %%
f, axs = plt.subplots(1, 2, figsize=(8,3), constrained_layout=True)
f.patch.set_facecolor('white')
axs[0].hist(np_epochs_all[~np.isnan(np_epochs_all.astype(float))],
         bins=30, edgecolor='black')
# axs[1].hist(np_epochs_all_folded[~np.isnan(np_epochs_all_folded.astype(float))],
#          bins=30, edgecolor='black')
axs[1].hist(np_epochs_all_folded[qualifies_by_epoch],
         bins=30, edgecolor='black')
axs[0].set_title('All cells together')
axs[1].set_title('Cells separated')
axs[0].set_ylabel('Number of models')
f.supxlabel('Number of training epochs')

# %%