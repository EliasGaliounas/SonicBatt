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

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell_ml')
data_path = os.path.join(study_path, 'Raw Data')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')
models_path = os.path.join(study_path, 'Models', 'Classification')

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

n_epochs = 8000

# %%
# %%
# ML helper functions
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

def schedule2(epoch):
    start_lr = 0.001
    end_lr = start_lr/5
    rampup_epochs = 300
    n_epochs = 8000
    return lr_linear_decay(epoch, start_lr, end_lr, rampup_epochs, n_epochs)

def ffnn_model(name, input_shape, depth, nodes_per_layer, normaliser):
    # Be very careful to avoid building on top of previous keras objects.
    keras.backend.clear_session()
    model = keras.Sequential(name=name)
    model.add(tf.keras.Input(shape=input_shape))
    model.add(normaliser)
    for i in range(depth):
        model.add(layers.Dense(nodes_per_layer, activation="relu", name=f"Layer_{i+1}"))
    # Specific to classification:
    model.add(layers.Dense(7, name="Layer_logits"))
    # model.add(layers.Softmax(name="Layer_softmax"))
    return(model)

def train_eval_model(representation, X_train, y_train, X_val, y_val, X_test, savedir, model_name):
    if 'model' in globals():
        del(model)
    repr_type = representations[representation]['repr_type']
    if repr_type == 'DL':
        normaliser = layers.Normalization(axis=-1, name='normaliser')
        normaliser.adapt(X_train)
        depth = representations[representation]['depth']
        nodes_per_layer = representations[representation]['nodes_per_layer']
        model = ffnn_model(name=model_name, input_shape=X_train.shape[1],
            depth=depth, nodes_per_layer=nodes_per_layer, normaliser=normaliser)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        # Make callback to adjust the learning rate
        schedule = representations[representation]['schedule']
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
        ax.set_ylabel('Loss\n(Sparse Categorical Cross Entropy)')
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
    'ffnn_1x2_sch2': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 2, 'schedule': schedule2},
    'ffnn_1x5_sch2': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 5, 'schedule': schedule2},
    'ffnn_1x10_sch2': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 10, 'schedule': schedule2},
    'ffnn_2x2_sch2': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 2, 'schedule': schedule2},
    'ffnn_2x5_sch2': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 5, 'schedule': schedule2},
    'ffnn_2x10_sch2': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 10, 'schedule': schedule2},
}

data_configs = {
    'A': [('peak_tofs', '8'), ('peak_heights', '8')],
    'B': ['peak_tofs'],
    'C': ['peak_tofs', 'peak_heights'],
    'D': ['acoustics'],
}

# %%
for data_config in data_configs.keys():
    feature_columns = data_configs[data_config]
    for representation in representations.keys():
        model_name = '{}_{}'.format(representation, data_config)
        train_indices = cells_together_split['train']
        val_indices = cells_together_split['val']
        test_indices = cells_together_split['test']
        X_train = df.loc[train_indices, feature_columns].to_numpy()
        y_train = df.loc[train_indices, label_column].to_numpy().reshape(-1,1)
        X_val = df.loc[val_indices, feature_columns].to_numpy()
        y_val = df.loc[val_indices, label_column].to_numpy().reshape(-1,1)
        X_test = df.loc[test_indices, feature_columns].to_numpy()
        y_test = df.loc[test_indices, label_column].to_numpy().reshape(-1,1)

        if 'model' in globals():
            del(model)
        savedir = os.path.join(models_path, model_name)
        pred_val, pred_test = train_eval_model(
            representation, X_train, y_train, X_val, y_val, X_test,
            savedir=savedir, model_name=model_name)

# %%
