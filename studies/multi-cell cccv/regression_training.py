# %%
from SonicBatt import utils
import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from keras import layers

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell cccv')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')
models_path = os.path.join(study_path, 'Regression_2')

parquet_filename = 'signals_peaks_fft.parquet'
parquet_filepath = os.path.join(ancillary_data_path, parquet_filename)
df = pd.read_parquet(parquet_filepath)
# Get rid of the invariable parts of the acoustic signals
df = df.drop(columns = [('acoustics', str(i)) for i in range(758)])

label_column = ('cycling', 'V(V)')

with open(os.path.join(ancillary_data_path,'cells_together_split.json'), 'r') as fp:
    cells_together_split = json.load(fp)
with open(os.path.join(ancillary_data_path,'cells_separated_splits.json'), 'r') as fp:
    cells_separated_splits = json.load(fp)

n_epochs = 8000

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

def schedule1(epoch):
    """
    No decay
    """
    start_lr = 0.001
    end_lr = 0.001
    rampup_epochs = 300
    n_epochs = 8000
    return lr_linear_decay(epoch, start_lr, end_lr, rampup_epochs, n_epochs)

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
    model.add(layers.Dense(1, name="Layer_last"))
    return(model)

def train_eval_model(representation, X_train, y_train, X_val, y_val, X_test, savedir, model_name):

    # If a tf dataset pipeline is to be used:
    # train_dataset  = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # val_dataset  = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # train_dataset =  train_dataset.cache('cache_file').prefetch(tf.data.AUTOTUNE)
    # val_dataset =  val_dataset.cache().prefetch(tf.data.AUTOTUNE)

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
            loss='mean_absolute_error',
        )
        # Make callback to adjust the learning rate
        schedule = representations[representation]['schedule']
        lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule)
        fitparams = {'epochs':n_epochs,'batch_size':64,'verbose':2, 'callbacks':[lr_callback, earlystop_callback]}
        #
        history = model.fit(X_train, y_train, 
                            validation_data=(X_val, y_val), **fitparams)
        # If a tf dataset pipeline is to be used:
        # history = model.fit(x = train_dataset, validation_data=val_dataset,
        #                      **fitparams)

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
    elif repr_type == 'ML':
        scaler = StandardScaler()
        scaler.fit(X_train)
        # save the scaler and transform X_train
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        save_dir_scaler = os.path.join(savedir, 'scaler.sav')
        with open(save_dir_scaler, 'wb') as fa:
            pickle.dump(scaler, fa)
        X_train_scaled = scaler.transform(X_train)
        # Reshape y_train to avoid scikit learn warnings
        y_train = y_train.ravel()

        if 'LinReg' in representation:
            model = LinearRegression()
        elif 'SVM' in representation:
            model = SVR()
        model.fit(X_train_scaled, y_train)
        save_dir_model = os.path.join(savedir, model_name) + '.sav'
        with open(save_dir_model, 'wb') as fa:
            pickle.dump(model, fa)
        return(_, _)
    
# %%
representations = {
    'LinReg': {'repr_type': 'ML'},
    'SVM': {'repr_type': 'ML'},
    # 'ffnn_1x10_sch1': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 10, 'schedule': schedule1},
    # 'ffnn_1x20_sch1': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 20, 'schedule': schedule1},
    # 'ffnn_1x50_sch1': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 50, 'schedule': schedule1},
    # 'ffnn_1x75_sch1': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 75, 'schedule': schedule1},
    # 'ffnn_1x100_sch1': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 100, 'schedule': schedule1},
    # 'ffnn_2x10_sch1': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 10, 'schedule': schedule1},
    # 'ffnn_2x20_sch1': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 20, 'schedule': schedule1},
    # 'ffnn_2x50_sch1': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 50, 'schedule': schedule1},
    # 'ffnn_2x75_sch1': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 75, 'schedule': schedule1},
    # 'ffnn_2x100_sch1': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 100, 'schedule': schedule1},

    # 'ffnn_1x10_sch2': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 10, 'schedule': schedule2},
    # 'ffnn_1x20_sch2': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 20, 'schedule': schedule2},
    # 'ffnn_1x50_sch2': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 50, 'schedule': schedule2},
    # 'ffnn_1x75_sch2': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 75, 'schedule': schedule2},
    # 'ffnn_1x100_sch2': {'repr_type': 'DL', 'depth': 1, 'nodes_per_layer': 100, 'schedule': schedule2},
    # 'ffnn_2x10_sch2': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 10, 'schedule': schedule2},
    # 'ffnn_2x20_sch2': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 20, 'schedule': schedule2},
    # 'ffnn_2x50_sch2': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 50, 'schedule': schedule2},
    # 'ffnn_2x75_sch2': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 75, 'schedule': schedule2},
    # 'ffnn_2x100_sch2': {'repr_type': 'DL', 'depth': 2, 'nodes_per_layer': 100, 'schedule': schedule2},
}

data_configs = {
    'A': [('peak_tofs', '8'), ('peak_heights', '8')],
    'B': ['peak_tofs'],
    'C': ['peak_tofs', 'peak_heights'],
    'D': ['acoustics'],
    'E': ['fft_magns'],
    'F': ['acoustics', 'fft_magns']
}

# %%
# cells_together
cell_config = 'cells_together'
data_config = 'C'
feature_columns = data_configs[data_config]
train_indices = cells_together_split['train']
val_indices = cells_together_split['val']
test_indices = cells_together_split['test']
for representation in representations.keys():
    model_name = '{}_{}'.format(representation, data_config)
    X_train = df.loc[train_indices, feature_columns].to_numpy()
    y_train = df.loc[train_indices, label_column].to_numpy().reshape(-1,1)
    X_val = df.loc[val_indices, feature_columns].to_numpy()
    y_val = df.loc[val_indices, label_column].to_numpy().reshape(-1,1)
    X_test = df.loc[test_indices, feature_columns].to_numpy()
    y_test = df.loc[test_indices, label_column].to_numpy().reshape(-1,1)

    if 'model' in globals():
        del(model)
    savedir = os.path.join(models_path, cell_config, model_name)
    pred_val, pred_test = train_eval_model(
        representation, X_train, y_train, X_val, y_val, X_test,
        savedir=savedir, model_name=model_name)

# %%
data_config = 'A'
feature_columns = data_configs[data_config]
representation = 'SVM'

model_name = '{}_{}'.format(representation, data_config)
savedir = os.path.join(models_path, cell_config, model_name)

#
save_dir_model = os.path.join(savedir, model_name) + '.sav'
with open(save_dir_model, 'rb') as fa:
    model = pickle.load(fa)
#
# print(len(model.coef_))
# print(model.intercept_)

n_support_vectors = model.support_vectors_.shape[0]

# Get the intercept value
intercept_value = model.intercept_

# Calculate the total number of parameters
n_parameters = n_support_vectors + 1  # +1 for the intercept

print("Number of support vectors:", n_support_vectors)
print("Intercept value:", intercept_value)
print("Total number of parameters:", n_parameters)

# %%

import numpy as np
save_dir_scaler = os.path.join(savedir, 'scaler.sav')
with open(save_dir_scaler, 'rb') as fa:
    scaler = pickle.load(fa)

X_test = df.loc[test_indices, feature_columns].to_numpy()
y_test = df.loc[test_indices, label_column].to_numpy().reshape(-1,1)

save_dir_model = os.path.join(savedir, model_name) + '.sav'
with open(save_dir_model, 'rb') as fa:
    model = pickle.load(fa)
pred_test = model.predict(scaler.transform(X_test))
mae_test = np.average(np.abs(y_test-pred_test))
print(round(mae_test,8))


# %%
# cells_separated
cell_config = 'cells_separated'
data_config = 'F'
feature_columns = data_configs[data_config]

for representation in representations.keys():
    for fold in cells_separated_splits.keys():
        print('Woking on fold: {}'.format(fold))
        model_name = '{}_{}_{}'.format(representation, data_config, fold)
        train_indices = cells_separated_splits[fold]['train']
        val_indices = cells_separated_splits[fold]['val']
        test_indices = cells_separated_splits[fold]['test']
        X_train = df.loc[train_indices, feature_columns].to_numpy()
        y_train = df.loc[train_indices, label_column].to_numpy().reshape(-1,1)
        X_val = df.loc[val_indices, feature_columns].to_numpy()
        y_val = df.loc[val_indices, label_column].to_numpy().reshape(-1,1)
        X_test = df.loc[test_indices, feature_columns].to_numpy()
        y_test = df.loc[test_indices, label_column].to_numpy().reshape(-1,1)

        if 'model' in globals():
            del(model)
        savedir = os.path.join(models_path, cell_config, model_name)
        _, _ = train_eval_model(
            representation, X_train, y_train, X_val, y_val, X_test,
            savedir=savedir, model_name=model_name)



# %%
