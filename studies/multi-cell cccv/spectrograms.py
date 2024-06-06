# %%
from SonicBatt import utils
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell cccv')
data_path = os.path.join(study_path, 'Raw Data')
visualistion_path = os.path.join(study_path, 'Visualisation')
ancillary_data_path = os.path.join(study_path, 'Ancillary Data')

database = pd.read_excel(os.path.join(data_path, 'database.xlsx'))
cell_aliases =  pd.read_excel(os.path.join(data_path, 'database.xlsx'),
                              sheet_name='cell_aliases')

parquet_filename = 'signals_peaks_fft.parquet'
parquet_filepath = os.path.join(ancillary_data_path, parquet_filename)
df = pd.read_parquet(parquet_filepath)
# Get rid of the invariable parts of the acoustic signals
df = df.drop(columns = [('acoustics', str(i)) for i in range(758)])

spectrograms = np.load(os.path.join(ancillary_data_path, 'spectrograms.npy'))

time_step=2.5e-03 # microseconds

# %%
def plot_spectrogram(spectrogram, ax, freqs, frame_step, frame_length):
    # Convert the frequencies to log scale and transpose, so that the time is.
    # Represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = utils.spectrogram_data_for_plot(spectrogram)
    # 
    X = (758*time_step + 
         (np.array(range(log_spec.shape[1]))*frame_step)*time_step + 
         round(frame_length/2)*time_step)
    # ax.pcolormesh(X, Y, log_spec)
    pcm = ax.pcolormesh(X, freqs, log_spec)
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label('Log (FFT Magn)\n(a.u.)')

time_step=2.5e-03 # microseconds
frame_length = 501
frame_step = 5
crop_freq = 21

freqs = np.fft.rfftfreq(n=frame_length, d = time_step)[1:crop_freq] #MHz

waveform = df.loc[0, 'acoustics']
spec = spectrograms[0]

f, axs = plt.subplots(2, figsize=(9, 6), sharex=True, constrained_layout=True)
timescale = np.arange(waveform.shape[0])*time_step + 758*time_step
axs[0].plot(timescale, waveform, color='tab:orange')
axs[0].set_title('Waveform (without invariant part)')
axs[0].set_xlim([758*time_step, 10])

plot_spectrogram(spec, axs[1], freqs, frame_step, frame_length)
axs[1].set_xlabel('ToF (μs)')
axs[0].set_ylabel('Acoustic amplitude (a.u.)')
axs[1].set_ylabel('Frequency (MHz)')
axs[1].set_title('Spectrogram')

f.align_ylabels()

# %%
# Animations
for cell in cell_aliases['cell_id'][3:]:
    filter_cell = df[('cycling', 'Cell_ID')] == cell
    cell_alias = cell_aliases.loc[cell_aliases['cell_id'] == cell, 'cell_alias'].iloc[0]
    cell_alias = 'Cell_{}'.format(cell_alias)
    filter_indices = np.array(df.loc[filter_cell].index)
    save_name = 'Spectrograms_{}'.format(cell_alias)
    utils.animate_spectrograms(
        df_cycling = df.loc[filter_cell, 'cycling'],
        signals = df.loc[filter_cell, 'acoustics'].to_numpy(),
        specs = spectrograms[filter_indices], title=cell_alias,
        save_dir=visualistion_path, save_name=save_name, fps=240)
    plt.close('all')
    plt.clf()

# %% 


# ------------------------------------------------
# Exploratory
# ------------------------------------------------


# %% 
waveform = df['acoustics'].iloc[0].to_numpy() # shape = (3242,)

def plot_spectrogram(spectrogram, ax, freqs, frame_step, frame_length):
    # Convert the frequencies to log scale and transpose, so that the time is.
    # Represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = utils.spectrogram_data_for_plot(spectrogram)
    # log_spec = log_spec[:,::-1]
    X = 758*time_step + (np.array(range(log_spec.shape[1]))*frame_step)*time_step + round(frame_length/2)*time_step
    # ax.pcolormesh(X, Y, log_spec)
    ax.pcolormesh(X, freqs, log_spec)

# 100, 200, 300, 400, 500
# 600, 700, 800, 900, 1000

frame_step = 5
crop_freq = 21

for frame_length in [501]:
    spectrogram = utils.make_spectrogram(
        waveform, frame_length=frame_length, frame_step=frame_step, 
        crop_freq=crop_freq, pad_end=False)
    
    freqs = np.fft.rfftfreq(n=frame_length, d = time_step)[1:crop_freq] #MHz

    f, axs = plt.subplots(2, figsize=(12, 8), sharex=True, constrained_layout=True)
    timescale = np.arange(waveform.shape[0])*time_step + 758*time_step
    axs[0].plot(timescale, waveform)
    axs[0].set_title('Waveform')
    axs[0].set_xlim([758*time_step, 10])

    plot_spectrogram(spectrogram.numpy(), axs[1], freqs, frame_step, frame_length)
    axs[1].set_title('Spectrogram, FL={}'.format(frame_length))
    plt.show()

# %%
