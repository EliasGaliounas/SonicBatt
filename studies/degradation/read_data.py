# %%
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import json
from SonicBatt import utils

root_dir = utils.root_dir()
data_path = os.path.join(root_dir, 'Raw Data')
database = pd.read_excel(os.path.join(data_path, 'database.xlsx'))
degr_tests = database.loc[database['test_type']=='degradation'].reset_index(drop=True)
cycles_completed = [int(i) for i in degr_tests['cycles_completed'].to_list()]
days_elapsed = degr_tests['days_elapsed'].to_list()
cycles_and_days = []
for i in range(len(cycles_completed)):
    cycles_and_days.append(
        '{} (day {})'.format(int(cycles_completed[i]), int(days_elapsed[i]))
        )

def json_to_custom_object(json_object, custom_object):
    if type(json_object) == dict:
        for key in Protocol_objects.keys():
            


# %%
for i in range(len(degr_tests)):
    test_id = degr_tests['test_id'].iloc[i]
    test_dir = os.path.join(data_path, test_id)
    # Protocol object
    file_name = '{}_Protocol_objects.json'.format(test_id)
    file_path = os.path.join(test_dir, file_name)
    with open(file_path, 'r') as f:
        json_string = f.read()
    Protocol_objects = json.loads(json_string)
    custom_object = utils.Protocol_custom_objects()

# test_id = 'EG_Ac109_231004_34'
# parquet_filename = '{}_acoustics_and_cycling.parquet'.format(test_id)
# parquet_filepath = os.path.join(data_path, test_id,parquet_filename)
# table = pq.read_table(parquet_filepath)
# df = table.to_pandas()

# %%
