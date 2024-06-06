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
study_path = os.path.join(root_dir, 'studies', 'multi-cell cccv')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')

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

# %%
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=50,
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

# %%
label_column = ('cycling', 'V(V)')

train_ind = cells_together_split['train']
val_ind = cells_together_split['val']
test_ind = cells_together_split['test']

train_specs = spectrograms[train_ind]
val_specs = spectrograms[val_ind]
test_specs = spectrograms[test_ind]

input_shape = spectrograms.shape[1:]
# Instantiate and fit the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization() # 
norm_layer.adapt(data=train_specs)

# %%
model = keras.Sequential([
    layers.Input(shape=input_shape),
    # Normalize.
    norm_layer,
    layers.Conv2D(16, 3, activation='relu'), #32, 3
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, activation='relu'), #64, 3
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(1),
])

# %%
n_epochs = 1000
lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule1)
fitparams = {'epochs':n_epochs,'batch_size':64,'verbose':2, 'callbacks':[lr_callback, earlystop_callback]}
#
y_train = df.loc[train_ind, label_column].to_numpy().reshape(-1,1)
y_val = df.loc[val_ind, label_column].to_numpy().reshape(-1,1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error',
)
history = model.fit(train_specs, y_train, 
                    validation_data=(val_specs, y_val), **fitparams)

# %%