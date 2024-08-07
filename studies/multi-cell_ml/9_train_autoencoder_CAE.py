
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
study_path = os.path.join(root_dir, 'studies', 'multi-cell_ml')
data_path = os.path.join(study_path, 'Raw Data')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')
unsupervised_models_path = os.path.join(study_path, 'Models', 'Unsupervised')

spectrograms = np.load(os.path.join(ancillary_data_path, 'spectrograms.npy'))

with open(os.path.join(ancillary_data_path,'cells_together_split.json'), 'r') as fp:
    cells_together_split = json.load(fp)

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
    start_lr = 0.0005
    end_lr = 0.001
    rampup_epochs = 300
    n_epochs = 8000
    return lr_linear_decay(epoch, start_lr, end_lr, rampup_epochs, n_epochs)

# %%
def get_scalers(X_train):
    min_values = np.min(X_train, axis=0)
    max_values = np.max(X_train, axis=0)
    # Avoid division by zero by replacing zeros in (max_values - min_values) with ones
    range_values = np.where(max_values - min_values == 0, 1, max_values - min_values)
    return(min_values, range_values)

def scale_data(data, min_values, range_values):
    data_scaled = (data - min_values) / range_values
    return(data_scaled)

def unscale_data(scaled_data, min_values, range_values):
    data_unscaled = (scaled_data * range_values) + min_values
    return(data_unscaled)

# %%
class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            # layers.Input(shape=(549, 20, 1)),
            layers.Input(shape=shape),
            layers.Dropout(0.2),
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(16, 3, activation='relu'),
            layers.Conv2D(8, 3, activation='relu'),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'), layers.Dropout(0.2),
            layers.Dense(512, activation='relu'), layers.Dropout(0.2),
            layers.Dense(256, activation='relu'), layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(latent_dim, activation='linear'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=latent_dim),
            layers.Dense(8, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'), layers.Dropout(0.2),
            layers.Dense(256, activation='relu'), layers.Dropout(0.2),
            layers.Dense(512, activation='relu'), layers.Dropout(0.2),
            layers.Dense(1024, activation='relu'),
            layers.Dense(543 * 14 * 8, activation='relu'),
            layers.Reshape((543, 14, 8)),
            layers.Conv2DTranspose(8, 3, activation='relu'),
            layers.Conv2DTranspose(16, 3, activation='relu'),
            layers.Conv2DTranspose(32, 3, activation='relu'),
            layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')   
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# %%
train_indices = cells_together_split['train']
val_indices = cells_together_split['val']
test_indices = cells_together_split['test']   
X_train = spectrograms[train_indices]
X_val = spectrograms[val_indices]
X_test = spectrograms[test_indices]

data_configs = {
    'G': 'spectrograms'
}
for data_config in data_configs.keys():
    keras.backend.clear_session()
    if 'model' in globals():
        del(model)

    min_values, range_values = get_scalers(X_train)
    X_train_scaled = scale_data(X_train, min_values, range_values)
    X_val_scaled = scale_data(X_val, min_values, range_values)

    shape = X_train.shape[1:]
    latent_dim = 2
    model = Autoencoder(latent_dim, shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',
    )

    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule1)
    fitparams = {'epochs':n_epochs,'batch_size':64,'verbose':2, 'callbacks':[lr_callback, earlystop_callback]}
    history = model.fit(
        X_train_scaled, X_train_scaled, 
        validation_data=(X_val_scaled, X_val_scaled),
        **fitparams)

    model_name = 'conv_autoencoder_sch2_{}_scaled'.format(data_config)
    savedir = os.path.join(unsupervised_models_path, model_name)
    save_dir_model = os.path.join(savedir, model_name)
    model.save(save_dir_model)
    save_dir_history = os.path.join(savedir, 'training_history.json')
    # Convert NumPy values to Python lists of floats
    for key in history.history:
        history.history[key] = [float(x) for x in history.history[key]]
    with open(save_dir_history, 'w') as fp:
        json.dump(history.history, fp)

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

# %%
