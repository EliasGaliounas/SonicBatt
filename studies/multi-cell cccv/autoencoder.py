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
from keras.models import Model

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell cccv')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')
unsupervised_models_path = os.path.join(study_path, 'Unsupervised')

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
n_epochs = 5

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

# %%
data_configs = {
    'B': ['peak_tofs'],
    'C': ['peak_tofs', 'peak_heights'],
    'D': ['acoustics'],
    'E': ['fft_magns'],
    'F': ['acoustics', 'fft_magns']
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
# https://www.tensorflow.org/tutorials/generative/autoencoder
class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(1000, activation='relu'),
            layers.Dense(800, activation='relu'),
            layers.Dense(600, activation='relu'),
            layers.Dense(400, activation='relu'),
            layers.Dense(200, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(25, activation='relu'),
            # layers.Dense(12, activation='relu'),
            # layers.Dense(4, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            # layers.Dense(4, activation='relu'),
            # layers.Dense(12, activation='relu'),
            layers.Dense(25, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(200, activation='relu'),
            layers.Dense(400, activation='relu'),
            layers.Dense(600, activation='relu'),
            layers.Dense(800, activation='relu'),
            layers.Dense(1000, activation='relu'),
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='relu'),
            layers.Reshape(shape)
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# %%
data_config = 'D'
feature_columns = data_configs[data_config]
X_train, y_train, X_val, y_val, X_test, y_test = config_data(data_config)

shape = X_train.shape[1:]
latent_dim = 10
autoencoder = Autoencoder(latent_dim, shape)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error',
)

lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule2)
fitparams = {'epochs':n_epochs,'batch_size':64,'verbose':2, 'callbacks':[lr_callback, earlystop_callback]}
autoencoder.fit(
    X_train, X_train, 
    validation_data=(X_val, X_val),
    **fitparams)

# %%
example_signal = X_train[0]

encoded_signals = autoencoder.encoder(X_train).numpy()
decoded_signals = autoencoder.decoder(encoded_signals).numpy()

f, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
# ax.plot(np.arange(len(example_signal)), example_signal, color='tab:blue')
ax.plot(np.arange(decoded_signals.shape[1]), decoded_signals[0], color='tab:orange')

# %%
x_values = X_test
y_values = y_test

encoded_test = autoencoder.encoder(x_values).numpy()
x1 = encoded_test[:, 0]
x2 = encoded_test[:, 1]

s=1
import matplotlib.colors as mcolors
colors = plt.cm.tab10.colors  # Get colors from tab10
cmap = mcolors.ListedColormap(colors[:7][::-1]) # Reverse them to look prettier

f, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
ax.scatter(x1, x2,
    c=y_values, edgecolor='none', alpha=0.5, cmap=cmap,s=s)

# %%