# %%
from SonicBatt import utils
import os
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell_ml')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')
models_path = os.path.join(study_path, 'Models', 'Regression')

parquet_filename = 'signals_peaks_fft.parquet'
parquet_filepath = os.path.join(ancillary_data_path, parquet_filename)
df = pd.read_parquet(parquet_filepath)
# Get rid of the invariable parts of the acoustic signals
df = df.drop(columns = [('acoustics', str(i)) for i in range(758)])

with open(os.path.join(ancillary_data_path,'cells_together_split.json'), 'r') as fp:
    cells_together_split = json.load(fp)
with open(os.path.join(ancillary_data_path,'cells_separated_splits.json'), 'r') as fp:
    cells_separated_splits = json.load(fp)

spectrograms = np.load(os.path.join(ancillary_data_path, 'spectrograms.npy'))
n_epochs = 8000

# %%
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=500,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)

def lr_linear_decay(epoch, start_lr, end_lr, rampup_epochs, n_epochs):
    if epoch < rampup_epochs:
        return start_lr
    else:
        decay = (start_lr - end_lr) / (n_epochs - rampup_epochs)
        return(start_lr - decay*(epoch-rampup_epochs))

def schedule1(epoch):
    start_lr = 0.0005
    end_lr = start_lr/5
    rampup_epochs = 300
    n_epochs = 8000
    return lr_linear_decay(epoch, start_lr, end_lr, rampup_epochs, n_epochs)

def schedule2(epoch):
    start_lr = 0.001
    end_lr = start_lr/5
    rampup_epochs = 300
    n_epochs = 8000
    return lr_linear_decay(epoch, start_lr, end_lr, rampup_epochs, n_epochs)

# %%
label_column = ('cycling', 'V(V)')

train_ind = cells_together_split['train']
val_ind = cells_together_split['val']
test_ind = cells_together_split['test']

# The ending '_specs' is to indicate that these are spectrograms
train_specs = spectrograms[train_ind] 
val_specs = spectrograms[val_ind]
test_specs = spectrograms[test_ind]

input_shape = spectrograms.shape[1:]
# Instantiate and fit the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization() # 
norm_layer.adapt(data=train_specs)

# %%
def cnn_model(name, input_shape, normaliser, n_filters_1=16, n_filters_2=32,
               dense_depth=1, dense_nodes_per_layer=10, dropout=False):
    # Be very careful to avoid building on top of previous keras objects.
    keras.backend.clear_session()
    model = keras.Sequential(name=name)
    model.add(tf.keras.Input(shape=input_shape))
    model.add(normaliser)
    model.add(layers.Conv2D(n_filters_1, 3, activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(n_filters_2, 3, activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    for i in range(dense_depth):
        model.add(layers.Dense(dense_nodes_per_layer, activation="relu", name=f"Layer_{i+1}"))
        if dropout:
            model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, name="Layer_last"))

    return(model)

def train_eval_model(representation, X_train, y_train, X_val, y_val, X_test, savedir, model_name):
    if 'model' in globals():
        del(model)
    normaliser = layers.Normalization(axis=-1, name='normaliser')
    normaliser.adapt(X_train)

    n_filters_1 = representations[representation]['n_filters_1']
    n_filters_2 = representations[representation]['n_filters_2']
    dense_depth = representations[representation]['dense_depth']
    dense_nodes_per_layer = representations[representation]['dense_nodes_per_layer']
    schedule = representations[representation]['schedule']
    dropout = representations[representation]['dropout']

    model = cnn_model(model_name, input_shape=X_train.shape[1:], normaliser=normaliser,
                    n_filters_1=n_filters_1, n_filters_2=n_filters_2,
                    dense_depth=dense_depth, dense_nodes_per_layer=dense_nodes_per_layer,
                    dropout=dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',
    )

    # Make callback to adjust the learning rate
    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule)

    fitparams = {'epochs':n_epochs,'batch_size':64,'verbose':2, 'callbacks':[lr_callback, earlystop_callback]}
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), **fitparams)
    
    # Save model and history
    save_dir_model = os.path.join(savedir, model_name) + '.h5'
    model.save(save_dir_model)
    save_dir_history = os.path.join(savedir, 'training_history.json')
    # Convert NumPy values to Python lists of floats
    for key in history.history:
        history.history[key] = [float(x) for x in history.history[key]]
    with open(save_dir_history, 'w') as fp:
        json.dump(history.history, fp)
    # Plot training history
    # -------------------------------------------------------
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    f, ax = plt.subplots(1,1, figsize=(6,4), constrained_layout=True)
    start = 10
    ax.plot(epochs[start:], loss[start:], c='tab:blue', marker='o', markersize=1, label='Training loss')
    ax.plot(epochs[start:], val_loss[start:], c='tab:orange', marker='o', markersize=1, label='Validation loss')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('')
    ax.set_xlim(start, len(epochs))
    f.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig_filename = os.path.join(savedir, model_name)
    f.savefig(fig_filename, bbox_inches='tight')
    # Assess
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    
    return(pred_val, pred_test)

# %%
representations = {
    # No dropout
    # ------------------------------------------------------------------------------------------
    'cnn_filters_8_16_dense_1x50_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 1, 'dense_nodes_per_layer': 50,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_16_32_dense_1x50_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 1, 'dense_nodes_per_layer': 50,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_32_64_dense_1x50_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 1, 'dense_nodes_per_layer': 50,
        'schedule': schedule2, 'dropout': False}, 
    
    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_1x100_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 1, 'dense_nodes_per_layer': 100,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_16_32_dense_1x100_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 1, 'dense_nodes_per_layer': 100,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_32_64_dense_1x100_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 1, 'dense_nodes_per_layer': 100,
        'schedule': schedule2, 'dropout': False}, 
    
    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_2x50_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 2, 'dense_nodes_per_layer': 50,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_16_32_dense_2x50_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 2, 'dense_nodes_per_layer': 50,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_32_64_dense_2x50_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 2, 'dense_nodes_per_layer': 50,
        'schedule': schedule2, 'dropout': False}, 
    
    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_2x100_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 2, 'dense_nodes_per_layer': 100,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_16_32_dense_2x100_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 2, 'dense_nodes_per_layer': 100,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_32_64_dense_2x100_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 2, 'dense_nodes_per_layer': 100,
        'schedule': schedule2, 'dropout': False},

    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_3x50_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 3, 'dense_nodes_per_layer': 50,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_16_32_dense_3x50_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 3, 'dense_nodes_per_layer': 50,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_32_64_dense_3x50_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 3, 'dense_nodes_per_layer': 50,
        'schedule': schedule2, 'dropout': False}, 
    
    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_3x100_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 3, 'dense_nodes_per_layer': 100,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_16_32_dense_3x100_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 3, 'dense_nodes_per_layer': 100,
        'schedule': schedule2, 'dropout': False},

    'cnn_filters_32_64_dense_3x100_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 3, 'dense_nodes_per_layer': 100,
        'schedule': schedule2, 'dropout': False},
    
    # ------------------------------------------------------------------------------------------
    # With dropout
    # ------------------------------------------------------------------------------------------
    'cnn_filters_8_16_dense_1x50_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 1, 'dense_nodes_per_layer': 50,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_16_32_dense_1x50_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 1, 'dense_nodes_per_layer': 50,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_32_64_dense_1x50_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 1, 'dense_nodes_per_layer': 50,
        'schedule': schedule1, 'dropout': True}, 
    
    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_1x100_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 1, 'dense_nodes_per_layer': 100,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_16_32_dense_1x100_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 1, 'dense_nodes_per_layer': 100,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_32_64_dense_1x100_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 1, 'dense_nodes_per_layer': 100,
        'schedule': schedule1, 'dropout': True}, 
    
    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_2x50_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 2, 'dense_nodes_per_layer': 50,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_16_32_dense_2x50_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 2, 'dense_nodes_per_layer': 50,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_32_64_dense_2x50_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 2, 'dense_nodes_per_layer': 50,
        'schedule': schedule1, 'dropout': True}, 
    
    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_2x100_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 2, 'dense_nodes_per_layer': 100,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_16_32_dense_2x100_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 2, 'dense_nodes_per_layer': 100,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_32_64_dense_2x100_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 2, 'dense_nodes_per_layer': 100,
        'schedule': schedule1, 'dropout': True},

    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_3x50_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 3, 'dense_nodes_per_layer': 50,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_16_32_dense_3x50_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 3, 'dense_nodes_per_layer': 50,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_32_64_dense_3x50_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 3, 'dense_nodes_per_layer': 50,
        'schedule': schedule1, 'dropout': True}, 
    
    # ------------------------------------------------------------------------------------------

    'cnn_filters_8_16_dense_3x100_sch2': {
        'n_filters_1': 8, 'n_filters_2': 16, 'dense_depth': 3, 'dense_nodes_per_layer': 100,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_16_32_dense_3x100_sch2': {
        'n_filters_1': 16, 'n_filters_2': 32, 'dense_depth': 3, 'dense_nodes_per_layer': 100,
        'schedule': schedule1, 'dropout': True},

    'cnn_filters_32_64_dense_3x100_sch2': {
        'n_filters_1': 32, 'n_filters_2': 64, 'dense_depth': 3, 'dense_nodes_per_layer': 100,
        'schedule': schedule1, 'dropout': True},        
}

# %%
# cells_together
# the variable 'cell_config', defined in the loop, will ensure that results go into appropriate folders
train_indices = cells_together_split['train']
val_indices = cells_together_split['val']
test_indices = cells_together_split['test']
data_config = 'G'

X_train = spectrograms[train_indices]
X_val = spectrograms[val_indices]
X_test = spectrograms[test_indices]
y_train = df.loc[train_indices, label_column].to_numpy().reshape(-1,1)
y_val = df.loc[val_indices, label_column].to_numpy().reshape(-1,1)
y_test = df.loc[test_indices, label_column].to_numpy().reshape(-1,1)

for representation in representations.keys():
    if 'model' in globals():
        del(model)
    dropout = representations[representation]['dropout']
    if dropout == False:
        cell_config = 'cells_together'
    else:
        cell_config = 'cells_together_dropout'
    model_name = '{}_{}'.format(representation, data_config)
    savedir = os.path.join(models_path, cell_config, model_name)
    _, _ = train_eval_model(
        representation, X_train, y_train, X_val, y_val, X_test,
        savedir=savedir, model_name=model_name)    

# %%
# cells_separated
# the variable 'cell_config', defined in the loop, will ensure that results go into appropriate folders
for fold in cells_separated_splits.keys():
    print('Working on fold: {}'.format(fold))
    train_indices = cells_separated_splits[fold]['train']
    val_indices = cells_separated_splits[fold]['val']
    test_indices = cells_separated_splits[fold]['test']

    X_train = spectrograms[train_indices]
    X_val = spectrograms[val_indices]
    X_test = spectrograms[test_indices]
    y_train = df.loc[train_indices, label_column].to_numpy().reshape(-1,1)
    y_val = df.loc[val_indices, label_column].to_numpy().reshape(-1,1)
    y_test = df.loc[test_indices, label_column].to_numpy().reshape(-1,1)

    for representation in representations.keys():
        if 'model' in globals():
            del(model)
        dropout = representations[representation]['dropout']
        if dropout == False:
            cell_config = 'cells_separated'
        else:
            cell_config = 'cells_separated_dropout'
        model_name = '{}_{}_{}'.format(representation, data_config, fold)
        savedir = os.path.join(models_path, cell_config, model_name)
        _, _ = train_eval_model(
            representation, X_train, y_train, X_val, y_val, X_test,
            savedir=savedir, model_name=model_name)            

# %%