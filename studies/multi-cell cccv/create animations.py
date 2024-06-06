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
freqs_1d = np.loadtxt(os.path.join(ancillary_data_path, 'frequencies.txt'))

parquet_filename = 'signals_peaks_fft.parquet'
parquet_filepath = os.path.join(ancillary_data_path, parquet_filename)
df = pd.read_parquet(parquet_filepath)

# %%
for test_id in database['test_id'].iloc[:4]:
    print('Working on {}'.format(test_id))
    cell_alias = cell_aliases.loc[cell_aliases['cell_id']==test_id[:6], 'cell_alias'].iloc[0]
    cell_alias = 'Cell_{}'.format(cell_alias)
    print('The alias for this cell is: Cell {}'.format(cell_alias))
    filter = df[('cycling', 'Cell_ID')] == test_id[:6]
    temp_df = df.loc[filter]

    acoustic_signals = temp_df['acoustics'].to_numpy()
    df_cycling = temp_df['cycling'].reset_index(drop=True)
    peak_heights = temp_df['peak_heights'].to_numpy()
    peak_tofs = temp_df['peak_tofs'].to_numpy()
    fft_magn = temp_df['fft_magns'].to_numpy()
    fft_magn = fft_magn[:, :150]
    freqs = freqs_1d[1:151]

    ani_filename = 'Animation_{}'.format(cell_alias)

    if not os.path.exists(visualistion_path):
        os.makedirs(visualistion_path)

    utils.animate_signals(
        df_cycling=df_cycling, signals=acoustic_signals,
        peak_heights=peak_heights, peak_tofs=peak_tofs,
        fft_magns = fft_magn, freqs_MHz = freqs,
        crop_ind = 758, fps=240, title = cell_alias,
        save_dir = visualistion_path, save_name=ani_filename)
    plt.close('all')
    plt.clf()

  # %%
