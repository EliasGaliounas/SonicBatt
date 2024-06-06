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
filter = database['discarded'] == 'N'
selected_cells = database.loc[filter, 'cell_id'].to_list()
df_cell_aliases =  pd.read_excel(os.path.join(data_path, 'database.xlsx'),
                              sheet_name='cell_aliases')
cell_aliases = {}
for _, row in df_cell_aliases.iterrows():
    cell_aliases[row['cell_id']] = row['cell_alias']

parquet_filename = 'signals_peaks_fft.parquet'
parquet_filepath = os.path.join(ancillary_data_path, parquet_filename)
df = pd.read_parquet(parquet_filepath)

frequencies_filename = 'frequencies.txt'
frequencies_filepath = os.path.join(ancillary_data_path, frequencies_filename)
freqs_id = np.loadtxt(frequencies_filepath)

# %%
from importlib import reload
reload(utils)

c_rates = [0.2, 0.5, 1]
# Time domain
utils.multi_cell_plot(df, selected_cells, cell_aliases, x_quantity = 'Q(mAh)',
                      c_rates = c_rates, domain = 'time', relative_peaks = True)

utils.multi_cell_plot(df, selected_cells, cell_aliases, x_quantity = 'V(V)',
                      c_rates = c_rates, domain = 'time', relative_peaks = True,
                      xlims = (2.7, 4.25))

# Frequency domain
# Frequency ndex closest to 2.25 MHz: 18
# Frequency ndex closest to 5 MHz: 41
utils.multi_cell_plot(df, selected_cells, cell_aliases, x_quantity = 'Q(mAh)',
                      c_rates = c_rates, domain = 'freq', freqs_1d = freqs_id, freq_ind_pair = (18, 41))

utils.multi_cell_plot(df, selected_cells, cell_aliases, x_quantity = 'V(V)',
                      c_rates = c_rates, domain = 'freq', freqs_1d = freqs_id, freq_ind_pair = (18, 41),
                      xlims = (2.7, 4.25))

# %%
for cell_id in selected_cells:
    f, axs = plt.subplots(6, 1, figsize=(11,7), dpi=300,
        height_ratios = [1.5,2.5,1.5,2.5,1,1.5], sharex=True, #height_ratios = [2,3,1.5,2.5,1]
        constrained_layout=True)
    f.patch.set_facecolor('white')
    filter = df[('cycling', 'Cell_ID')] == cell_id
    utils.plot_cycling_data(df.loc[filter, 'cycling'], df.loc[filter,'peak_tofs'],
                      f, axs)
    f.suptitle('Cell {}'.format(cell_aliases[cell_id]))
    save_filename = '210_cccv_original_timeseries_{}.pdf'.format(cell_id)
    # save_figure(f, save_filename)
    break
