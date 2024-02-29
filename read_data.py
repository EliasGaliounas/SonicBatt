# %%
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import json
data_path = 'C:\\Users\elias\C_BatteryAcousticStudies2\Raw data'

# %%
test_id = 'EG_Ac109_231004_34'
parquet_filename = '{}_acoustics_and_cycling.parquet'.format(test_id)
parquet_filepath = os.path.join(data_path, test_id,parquet_filename)
table = pq.read_table(parquet_filepath)
df = table.to_pandas()

# %%
