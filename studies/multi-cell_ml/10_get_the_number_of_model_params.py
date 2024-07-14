""""
The script shows how to get the number of tuneable parameters from different models
"""

# %%
from SonicBatt import utils
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell_ml')
data_path = os.path.join(study_path, 'Raw Data')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')
unsupervised_models_path = os.path.join(study_path, 'Models', 'Unsupervised')
autoencoders_path = os.path.join(unsupervised_models_path, 'Autoencoders')

# %%
# Classical models - Regression
models_path = os.path.join(study_path, 'Models', 'Regression')

cell_config = 'cells_together'
for representation in ['LinReg', 'SVM']:
    for data_config in ['A', 'B', 'C', 'D', 'E', 'F']:
        model_name = '{}_{}'.format(representation, data_config)
        model_dir = os.path.join(models_path, cell_config, model_name)
        with open(os.path.join(model_dir, model_name + '.sav'), 'rb') as fa:
            model = pickle.load(fa)
        if representation == 'LinReg':
            print('{}: {} coefficients'.format(model_name, len(model.coef_)+1)) # +1 is for the model intercept
        elif representation == 'SVM':
            print('{}: {} support vectors'.format(model_name, model.support_vectors_.shape[0]))
  
cell_config = 'cells_separated'
for representation in ['LinReg', 'SVM']:
    for data_config in ['A', 'B', 'C', 'D', 'E', 'F']:
        for fold in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']:
            model_name = '{}_{}_{}'.format(representation, data_config, fold)
            model_dir = os.path.join(models_path, cell_config, model_name)
            with open(os.path.join(model_dir, model_name + '.sav'), 'rb') as fa:
                model = pickle.load(fa)
            if representation == 'LinReg':
                print('{}: {} coefficients'.format(model_name, len(model.coef_)+1)) # +1 is for the model intercept
            elif representation == 'SVM':
                print('{}: {} support vectors'.format(model_name, model.support_vectors_.shape[0]))

# %%
# FNN models - Regression
models_path = os.path.join(study_path, 'Models', 'Regression')

representations = [
    'ffnn_1x10_sch2', 'ffnn_1x20_sch2', 'ffnn_1x50_sch2', 'ffnn_1x75_sch2', 'ffnn_1x100_sch2',
    'ffnn_2x10_sch2', 'ffnn_2x20_sch2', 'ffnn_2x50_sch2', 'ffnn_2x75_sch2', 'ffnn_2x100_sch2',
    'ffnn_3x10_sch2', 'ffnn_3x20_sch2', 'ffnn_3x50_sch2', 'ffnn_3x75_sch2', 'ffnn_3x100_sch2',

    'cnn_filters_8_16_dense_1x50_sch2', 'cnn_filters_16_32_dense_1x50_sch2', 'cnn_filters_32_64_dense_1x50_sch2',
    'cnn_filters_8_16_dense_1x100_sch2', 'cnn_filters_16_32_dense_1x100_sch2', 'cnn_filters_32_64_dense_1x100_sch2',
    
    'cnn_filters_8_16_dense_2x50_sch2', 'cnn_filters_16_32_dense_2x50_sch2', 'cnn_filters_32_64_dense_2x50_sch2',
    'cnn_filters_8_16_dense_2x100_sch2', 'cnn_filters_16_32_dense_2x100_sch2', 'cnn_filters_32_64_dense_2x100_sch2',

    'cnn_filters_8_16_dense_3x50_sch2', 'cnn_filters_16_32_dense_3x50_sch2', 'cnn_filters_32_64_dense_3x50_sch2',
    'cnn_filters_8_16_dense_3x100_sch2', 'cnn_filters_16_32_dense_3x100_sch2', 'cnn_filters_32_64_dense_3x100_sch2',
]

cell_config = 'cells_together'
for representation in representations:
    if 'cnn_filters' not in representation:
        for data_config in ['B', 'C', 'D', 'E', 'F']:
            model_name = '{}_{}'.format(representation, data_config)
            model_dir = os.path.join(models_path, cell_config, model_name)
            model = load_model(os.path.join(model_dir, model_name + '.h5'))
            trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            print('{}: {} trainable parameters'.format(model_name, trainable_params))
    else:
        data_config = 'G'
        model_name = '{}_{}'.format(representation, data_config)
        model_dir = os.path.join(models_path, cell_config, model_name)
        model = load_model(os.path.join(model_dir, model_name + '.h5'))
        trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        print('{}: {} trainable parameters'.format(model_name, trainable_params))        

cell_config = 'cells_separated'
fold = 'F1' # it's the same for all folds
for representation in representations:
    if 'cnn_filters' not in representation:
        for data_config in ['B', 'C', 'D', 'E', 'F']:
            model_name = '{}_{}_{}'.format(representation, data_config, fold)
            model_dir = os.path.join(models_path, cell_config, model_name)
            model = load_model(os.path.join(model_dir, model_name + '.h5'))
            trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            print('{}: {} trainable parameters'.format(model_name, trainable_params))
    else:
        data_config = 'G'
        model_name = '{}_{}_{}'.format(representation, data_config, fold)
        model_dir = os.path.join(models_path, cell_config, model_name)
        model = load_model(os.path.join(model_dir, model_name + '.h5'))
        trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        print('{}: {} trainable parameters'.format(model_name, trainable_params))        

# %%
# FNN models - Classification
models_path = os.path.join(study_path, 'Models', 'Classification')

representations = [
    'ffnn_1x2_sch2', 'ffnn_1x5_sch2', 'ffnn_1x10_sch2',
    'ffnn_2x2_sch2', 'ffnn_2x5_sch2', 'ffnn_2x10_sch2'
]

for representation in representations:
    for data_config in ['A', 'B', 'C', 'D']:
        model_name = '{}_{}'.format(representation, data_config)
        model_dir = os.path.join(models_path, model_name)
        model = load_model(os.path.join(model_dir, model_name + '.h5'))
        trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        print('{}: {} trainable parameters'.format(model_name, trainable_params))        

# %%
# Autoencoders
model_names = [
    'autoencoder_sch2_D_scaled',
    'autoencoder_sch2_F_scaled',
    'autoencoder_sch2_G_scaled',
    'conv_autoencoder_sch2_G_scaled'
]

# To see parameters in individual layers:
for model_name in model_names:
    model_dir = os.path.join(autoencoders_path, model_name)
    model = load_model(os.path.join(model_dir, model_name))
    print('Working on: {}'.format(model_name))
    for layer in model.encoder.layers:
        config = layer.get_config()
        layer_type = layer.__class__.__name__
        activation = config.get('activation', 'None')
        units = config.get('units', 'N/A')  # Get the number of units if available
        # if layer_type in ['Conv2D', 'Conv1D', 'Conv3D', 'Conv2DTranspose']:
        #         filters = config.get('filters', 'N/A')
        #         units  = filters
        print(f'Layer: {layer_type}, Units: {units}, Activation: {activation}')
    print('')

# To get the model summary showing all parameters:
for model_name in model_names:
    model_dir = os.path.join(autoencoders_path, model_name)
    model = load_model(os.path.join(model_dir, model_name))
    print('This is for: {}'.format(model_name))
    print(model.summary())
    print('')

# %%





