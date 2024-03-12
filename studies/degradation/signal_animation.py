# %%
from SonicBatt import utils
import os
import pandas as pd
import pyarrow.parquet as pq


root_dir = utils.root_dir()
study_path = os.path.join(root_dir, 'studies', 'degradation')
data_path = os.path.join(study_path, 'Raw Data')
visualistion_path = os.path.join(study_path, 'Visualisation')

database = pd.read_excel(os.path.join(data_path, 'database.xlsx'))
rate_tests = database.loc[database['test_type']=='multi_c_rate'].reset_index(drop=True)

# From the multi-c-rate study files select the file corresponding to the rate of 1C
selected_c_rate = 1
filter = rate_tests['c_rate'] == selected_c_rate
selected_test_id = rate_tests.loc[filter, 'test_id'].iloc[0]

# %%
# Load complete acoustic waveforms:
parquet_filename = '{}_acoustics_and_cycling.parquet'.format(selected_test_id)
parquet_filepath = os.path.join(data_path, selected_test_id, parquet_filename)
df_signals = pq.read_table(parquet_filepath).to_pandas()
acoustic_signals = df_signals['acoustics']

# Find acoustic peaks
test_dir = os.path.join(data_path, selected_test_id)
df = utils.df_with_peaks(data_path, selected_test_id, passes=50)
df_cycling = df['cycling']
peak_heights = df['peak_heights']
peak_tofs = df['peak_tofs']

# %%
# Animate and save
acoustic_signals = df_signals['acoustics'].to_numpy()
df_cycling = df['cycling']
peak_heights = df['peak_heights'].to_numpy()
peak_tofs = df['peak_tofs'].to_numpy()

ani_filename = '1C_pulses_example'

if not os.path.exists(visualistion_path):
    os.makedirs(visualistion_path)

utils.animate_signals(
    df_cycling=df_cycling, signals=acoustic_signals,
    peak_heights=peak_heights, peak_tofs=peak_tofs, fps=240,
    save_dir = visualistion_path, save_name=ani_filename)

# %%
