# %%
from SonicBatt import utils
import os
import pandas as pd
import numpy as np
import json

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell cccv')
data_path = os.path.join(study_path, 'Raw Data')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')

database = pd.read_excel(os.path.join(data_path, 'database.xlsx'))
selected_cells = database.loc[database['discarded'] == 'N', 'cell_id'].to_list()

time_step = 2.5e-03 #microseconds

# %%
# Identify Acoustic Peaks
parquet_filename = 'signals_peaks_fft.parquet'
parquet_filepath = os.path.join(ancillary_data_path, parquet_filename)

if not os.path.exists(ancillary_data_path):
    os.makedirs(ancillary_data_path)
if not os.path.exists(parquet_filepath):
    for i, test_id in enumerate(database['test_id']):
        print('Working on {}'.format(test_id))
        test_dir = os.path.join(data_path, test_id)
        temp_df = utils.df_with_peaks(data_path, test_id, passes=50)
        if i == 0:
            df = temp_df
        else:
            df = pd.concat([df, temp_df], axis=0, ignore_index=True)

    # Filter out:
    # - Cells which are not in the selected cells
    # - The final part of each cell's dataset, if the cell was simply at rest
    filter_cells = df[('cycling','Cell_ID')].isin(selected_cells)
    filer_final_rest = df[('cycling','V(V)')].notna()
    df_shorter = df.loc[filter_cells & filer_final_rest].reset_index(drop=True)

    # Calculate FFT coefficients
    frequencies_filename = 'frequencies.txt'
    frequencies_filepath = os.path.join(ancillary_data_path, frequencies_filename)

    # if not os.path.exists(parquet_filepath):
    signals = df_shorter['acoustics'].to_numpy()
    crop_ind = int(4000/5) - np.argmax(
        np.flip(signals[:,:int(4000/5)], axis=1), axis=1).max()
    signals_cropped = signals[:,crop_ind:]

    if not os.path.exists(frequencies_filepath):
        freqs_1d = np.fft.rfftfreq(
            n = signals_cropped.shape[1], d = time_step) #MHz
        np.savetxt(frequencies_filepath, freqs_1d)

    fft_coeffs = np.fft.rfft(signals_cropped, axis=1)
    fft_magns = np.abs(fft_coeffs)
    fft_magns[:,0] = 0
    n_freqs = 301
    fft_headings = [str(i) for i in range(1, n_freqs)]
    df_fft_magns = pd.DataFrame(data = fft_magns[:,1:n_freqs], columns = fft_headings)

    df_final = pd.concat([
        df_shorter['cycling'], df_shorter['acoustics'], df_shorter['peak_heights'],
        df_shorter['peak_tofs'], df_fft_magns], axis = 1,
        keys = ['cycling', 'acoustics','peak_heights', 'peak_tofs',
                'fft_magns'])
    # Save the concatenated file
    df_final.to_parquet(parquet_filepath)
else:
    df_final = pd.read_parquet(parquet_filepath)

# %%
# Split the dataset for modelling
cells_separated_splits = {
    'F1': {'test_cell': 'EG_Ac9', 'val_cell': 'EG_Ac8',
           'train': None, 'val': None, 'test': None},
    'F2': {'test_cell': 'EG_Ac8', 'val_cell': 'EG_Ac6',
           'train': None, 'val': None, 'test': None},    
    'F3': {'test_cell': 'EG_Ac6', 'val_cell': 'EG_Ac5',
           'train': None, 'val': None, 'test': None},  
    'F4': {'test_cell': 'EG_Ac5', 'val_cell': 'EG_Ac4',
           'train': None, 'val': None, 'test': None}, 
    'F5': {'test_cell': 'EG_Ac4', 'val_cell': 'EG_Ac3',
           'train': None, 'val': None, 'test': None},
    'F6': {'test_cell': 'EG_Ac3', 'val_cell': 'EG_Ac2',
           'train': None, 'val': None, 'test': None},
    'F7': {'test_cell': 'EG_Ac2', 'val_cell': 'EG_Ac9',
           'train': None, 'val': None, 'test': None}, 
}
for fold in cells_separated_splits.keys():
    test_cell = cells_separated_splits[fold]['test_cell']
    val_cell = cells_separated_splits[fold]['val_cell']
    filter_test_cell = df_final[('cycling', 'Cell_ID')] == test_cell
    filter_val_cell = df_final[('cycling', 'Cell_ID')] == val_cell
    cells_separated_splits[fold]['train'] = (
        df_final.loc[~(filter_test_cell | filter_val_cell)]).index.to_numpy()
    cells_separated_splits[fold]['test'] = (
        df_final.loc[filter_test_cell]).index.to_numpy()
    cells_separated_splits[fold]['val'] = (
        df_final.loc[filter_val_cell]).index.to_numpy()    
    assert(True not in np.isin(
        cells_separated_splits[fold]['train'],
        cells_separated_splits[fold]['val']))
    assert(True not in np.isin(
        cells_separated_splits[fold]['train'],
        cells_separated_splits[fold]['test']))
    assert(True not in np.isin(
        cells_separated_splits[fold]['val'],
        cells_separated_splits[fold]['test']))
    # Convert to lists to save a json
    cells_separated_splits[fold]['train'] = cells_separated_splits[fold]['train'].tolist()
    cells_separated_splits[fold]['test'] = cells_separated_splits[fold]['test'].tolist()
    cells_separated_splits[fold]['val'] = cells_separated_splits[fold]['val'].tolist()

np.random.seed(42)
cells_together_split = {'train': None, 'val': None, 'test': None}
#
indices_all = df.index.to_numpy().copy()
# !!! Instead of what didn't work --> #df.copy(deep=True).index.to_numpy()
#
n_train = round(0.6 * len(indices_all))
n_val = round(0.2 * len(indices_all))
np.random.shuffle(indices_all)
cells_together_split['train'] = indices_all[:n_train]
cells_together_split['val'] = indices_all[n_train:(n_train+n_val)]
cells_together_split['test'] = indices_all[(n_train + n_val):]
assert(True not in np.isin(
    cells_together_split['train'],
    cells_together_split['val']))
assert(True not in np.isin(
    cells_together_split['train'],
    cells_together_split['test']))
assert(True not in np.isin(
    cells_together_split['val'],
    cells_together_split['test']))
# Convert to lists to save a json
cells_together_split['train'] = cells_together_split['train'].tolist()
cells_together_split['test'] = cells_together_split['test'].tolist()
cells_together_split['val'] = cells_together_split['val'].tolist()

# Save the indices for later use
with open(os.path.join(ancillary_data_path,'cells_together_split.json'), 'w') as fp:
    json.dump(cells_together_split, fp)
with open(os.path.join(ancillary_data_path,'cells_separated_splits.json'), 'w') as fp:
    json.dump(cells_separated_splits, fp)

# %% To load the indices from file:
with open(os.path.join(ancillary_data_path,'cells_together_split.json'), 'r') as fp:
    cells_together_split = json.load(fp)
with open(os.path.join(ancillary_data_path,'cells_separated_splits.json'), 'r') as fp:
    cells_separated_splits = json.load(fp)

# %%
# Construct spectrograms
# First remove the invariant part of the dataframes (the first 758 datapoints)
df = df_final.drop(columns = [('acoustics', str(i)) for i in range(758)])
time_step=2.5e-03 # microseconds
frame_length = 501
frame_step = 5
crop_freq = 21

def make_spec_dataset(waveforms):
    # Initialize a list to store spectrograms
    specs_list = []
    for i, waveform in enumerate(waveforms):
        spectrogram = utils.make_spectrogram(
            waveform, frame_length=frame_length, frame_step=frame_step, 
            crop_freq=crop_freq, pad_end=False)
        specs_list.append(spectrogram)
        # Print progress every 5000 iterations
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1} out of {len(waveforms)} waveforms")
    # Convert the list to a numpy array
    specs = np.array(specs_list)    
    return specs  

all_spectrograms = make_spec_dataset(df.loc[:, 'acoustics'].to_numpy())
save_dir = os.path.join(ancillary_data_path, 'spectrograms.npy')
np.save(save_dir, all_spectrograms)

# %%
# Save also th labels separately
y_values = df_final[('cycling', 'V(V)')].to_numpy()
save_dir = os.path.join(ancillary_data_path, 'V_labels.npy')
np.save(save_dir, y_values)

# %%