from SonicBatt import utils
import numpy as np
import pandas as pd
import pytest
import os
import matplotlib.pyplot as plt

def test_root_dir():
    my_dir = utils.root_dir()
    assert(my_dir != None)

    # Test the exception
    from unittest.mock import patch
    import subprocess
    with patch('subprocess.check_output') as mock_check_output:
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'git')
        my_dir = utils.root_dir()
        assert my_dir is None

def test_attribute_assignment():
    class_objects = [
        utils.Pulse(), utils.PulseSequence(),
        utils.Acoustic_Pulse(), utils.Acoustic_PulseSequence(),
        utils.EIS_object(), utils.EIS_sequence(),
        utils.Protocol_custom_objects(), utils.Acoustic_p2C()
    ]
    for obj in class_objects:
        for att in dir(obj):
            if not att.startswith('_'):
                setattr(obj, att, 0)
        for att in dir(obj):
            if not att.startswith('_'):
                assert getattr(obj, att) == 0
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

def test_signal_peaks(acoustic_data):
    df_acoust, _ = acoustic_data
    signals = df_acoust['acoustics'].to_numpy()
    peak_heights, peak_tofs = utils.signal_peaks(signals)
    assert(peak_heights.shape == peak_tofs.shape)

def test_df_with_peaks():
    data_path = os.path.join(os.getcwd(), 'tests')
    test_id = 'example_2'
    #
    for my_list in ([], [1,3,5], [1,3,5]):
        df = utils.df_with_peaks(data_path=data_path, test_id=test_id,
                slicing_indices=my_list, ignore_ind_list=my_list)
        first_row_headings = df.columns.levels[0]
        for item in ['cycling', 'acoustics', 'peak_heights', 'peak_tofs']:
            assert (item in first_row_headings)
    #
    n_peaks = 8
    df = utils.df_with_peaks(data_path=data_path, test_id=test_id,
                             n_peaks=n_peaks)
    first_row_headings = df.columns.levels[0]
    for item in ['cycling', 'acoustics', 'peak_heights', 'peak_tofs']:
        assert (item in first_row_headings)

def test_create_custom_object():
    data_path = os.path.join(os.getcwd(), 'tests')
    test_id = 'example_3'
    custom_object = utils.create_custom_object(test_id, data_path)
    assert (type(custom_object) == type(utils.Protocol_custom_objects()))

def test_animate_signals(acoustic_data):
    df_acoust, _ = acoustic_data
    signals = df_acoust['acoustics'].to_numpy()
    peak_heights, peak_tofs = utils.signal_peaks(signals) 
    # Only waveforms   
    ani = utils.animate_signals(df_cycling = df_acoust['cycling'],
                    signals = signals)
    assert (ani != None)
    # Waveforms and peaks
    ani = utils.animate_signals(df_cycling = df_acoust['cycling'],
            signals = signals, 
            peak_heights=peak_heights,
            peak_tofs=peak_tofs)
    assert (ani != None)

def test_animate_signals2(acoustic_data):
    # Frequency domain
    df_acoust, _ = acoustic_data
    signals = df_acoust['acoustics'].to_numpy()
    #
    time_step=2.5e-03
    freqs_MHz = np.fft.rfftfreq(signals.shape[1], time_step)
    fft_coeffs = np.fft.rfft(signals, axis=1)
    fft_magns = np.abs(fft_coeffs)
    ani = utils.animate_signals(df_cycling = df_acoust['cycling'],
                      signals = signals, fft_magns=fft_magns,
                      freqs_MHz=freqs_MHz)
    assert (ani != None)

def test_make_spectrogram(acoustic_data):
    df_acoust, _ = acoustic_data
    signals = df_acoust['acoustics'].to_numpy()
    s0 = signals[0]
    spectrogram = utils.make_spectrogram(s0)
    assert (spectrogram != None)

def test_spectrogram_data_for_plot(acoustic_data):
    df_acoust, _ = acoustic_data
    signals = df_acoust['acoustics'].to_numpy()
    s0 = signals[0]
    spectrogram = utils.make_spectrogram(s0)
    log_spec = utils.spectrogram_data_for_plot(spectrogram)
    assert (log_spec.size != 0)

def test_animate_spectrograms(acoustic_data):
    df_acoust, _ = acoustic_data
    signals = df_acoust['acoustics'].to_numpy()
    def make_spec_dataset(waveforms):
        # Initialize a list to store spectrograms
        specs_list = []
        for i, waveform in enumerate(signals):
            spectrogram = utils.make_spectrogram(waveform)
            specs_list.append(spectrogram)
            # Print progress every 5000 iterations
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1} out of {len(waveforms)} waveforms")
        # Convert the list to a numpy array
        specs = np.array(specs_list)
        return specs
    all_spectrograms = make_spec_dataset(signals)
    #
    ani = utils.animate_spectrograms(df_acoust['cycling'], signals, all_spectrograms)
    assert (ani != None)

def test_multi_cell_plot():
    data_path = os.path.join(os.getcwd(), 'tests')
    test_id = 'example_2'
    n_peaks = 8
    df = utils.df_with_peaks(data_path=data_path, test_id=test_id,
                             n_peaks=n_peaks)    
    # Make some changes to df for the sake of testing:
    df[('cycling','Cycle')] = 1
    df[('cycling','C-Rate')] = 1
    selected_cells = ['EG_Ac_34']
    cell_aliases = {'EG_Ac_34': 1}
    f, _ = utils.multi_cell_plot(df, selected_cells, cell_aliases, c_rates=[1], return_axes=True)
    assert f is not None
    assert isinstance(f, plt.Figure)
    # Now try the frequency domain
    # signals = df['acoustics'].to_numpy()
    # fft_coeffs = np.fft.rfft(signals, axis=1)
    # fft_magns = np.abs(fft_coeffs)
    # fft_magns[:,0] = 0
    # n_freqs = 301
    # fft_headings = [str(i) for i in range(1, n_freqs)]
    # df_fft_magns = pd.DataFrame(data = fft_magns[:,1:n_freqs], columns = fft_headings)
    # df = pd.concat([
    #     df['cycling'], df['acoustics'], df['peak_heights'],
    #     df['peak_tofs'], df_fft_magns], axis = 1,
    #     keys = ['cycling', 'acoustics','peak_heights', 'peak_tofs',
    #             'fft_magns'])



