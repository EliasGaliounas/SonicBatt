# %%
from SonicBatt import utils
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'multi-cell_ml')
data_path = os.path.join(study_path, 'Raw Data')
visualisation_path = os.path.join(study_path, 'Visualisation')
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
c_rates = [0.2, 0.5, 1]
# Time domain
save_filename = 'multicell_all_c_rates'
utils.multi_cell_plot(df, selected_cells, cell_aliases, x_quantity = 'Q(mAh)',
                      c_rates = c_rates, domain = 'time', relative_peaks = True,
                      save_filename = save_filename, visualisation_path = visualisation_path)

save_filename = 'multicell_0p2C'
f = utils.multi_cell_plot(df, selected_cells, cell_aliases, x_quantity = 'Q(mAh)',
                      c_rates = [0.2], domain = 'time', relative_peaks = True,
                      save_filename = save_filename, visualisation_path = visualisation_path)

save_filename = 'multicell_0p5C'
f = utils.multi_cell_plot(df, selected_cells, cell_aliases, x_quantity = 'Q(mAh)',
                      c_rates = [0.5], domain = 'time', relative_peaks = True,
                      save_filename = save_filename, visualisation_path = visualisation_path)

save_filename = 'multicell_1p0C'
f = utils.multi_cell_plot(df, selected_cells, cell_aliases, x_quantity = 'Q(mAh)',
                      c_rates = [1], domain = 'time', relative_peaks = True,
                      save_filename = save_filename, visualisation_path = visualisation_path)


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
    f, axs = plt.subplots(6, 1, figsize=(11,5), dpi=300,
        height_ratios = [1.5,1.5,1.5,1.5,1.5,2], sharex=True, #height_ratios = [2,3,1.5,2.5,1]
        constrained_layout=True)
    f.patch.set_facecolor('white')
    filter = df[('cycling', 'Cell_ID')] == cell_id
    utils.plot_cycling_data(df.loc[filter, 'cycling'], df.loc[filter,'peak_tofs'],
                      f, axs)
    f.suptitle('Cell {}'.format(cell_aliases[cell_id]))
    save_filename = '210_cccv_timeseries_cell_{}'.format(cell_aliases[cell_id])
    if cell_id == 'EG_Ac2':
        axs[2].set_ylim(22,28)
        axs[2].set_yticks([22,25,28])
        axs[3].set_yticks([0, 80, 160, 235])
        axs[3].set_ylim(0,235)
        axs[1].set_ylim(2.7, 4.35)
        axs[1].set_yticks([2.75, 4.2])
        axs[5].set_ylim(8.5, 9.1)
        axs[5].set_yticks([8.6, 8.8, 9])
        utils.save_figure(f, visualisation_path, save_filename, format='pdf')
    break

# %%