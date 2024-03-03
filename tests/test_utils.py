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
    for kernel_type in ['rectangular', 'triangular', 'gaussian']:
        for window_len in [11, 15]:
            for passes in [10, 15]:
                _, s0_smooth = utils.smooth_by_convolution(
                    s0, window_len, kernel_type, passes)
                assert ( len(s0) == len(s0_smooth) )

def test_attribute_assignment():
    class_objects = [
        utils.Pulse(), utils.PulseSequence(),
        utils.Acoustic_Pulse(), utils.Acoustic_PulseSequence(),
        utils.EIS_object(), utils.EIS_sequence(),
        utils.Protocol_custom_objects()
    ]
    for obj in class_objects:
        for att in dir(obj):
            if not att.startswith('_'):
                setattr(obj, att, 0)
        for att in dir(obj):
            if not att.startswith('_'):
                assert getattr(obj, att) == 0

def test_root_dir():
    assert(utils.root_dir() != None)