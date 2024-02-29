from SonicBatt import utils
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def acoustic_data():
    df_acoust = pd.read_parquet('tests/example_acoust.parquet')
    time_step = 2.5e-03 #microseconds
    time_range = df_acoust['acoustics'].shape[1]*time_step # microseconds
    time_axis = np.arange(time_step, time_range+time_step, time_step)
    return df_acoust, time_axis

def test_smooth_by_convolution(acoustic_data):
    df_acoust, _ = acoustic_data
    signals = df_acoust['acoustics'].to_numpy()
    s0 = signals[0]
    _, s0_smooth = utils.smooth_by_convolution(s0)
    assert ( len(s0) == len(s0_smooth) )

