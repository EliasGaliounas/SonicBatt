import numpy as np
import os
import json
import pandas as pd
import pyarrow.parquet as pq

def root_dir():
    import subprocess
    try:
        root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
        return root
    except subprocess.CalledProcessError:
        return None

class Pulse:
    """
    Contains information for single pulses.
    "C" indicates a step where current is not zero.
    "R" indicates a rest step.<br>
    """
    def __init__(self):
        # Index labelling:
        self.C_start_ind = None
        self.C_end_ind = None
        self.R_start_ind = None
        self.R_end_ind = None
        # Temperature info:
        self.C_temp_mean = None
        self.C_temp_stdev = None
        self.C_temp_max = None
        self.C_temp_min = None
        self.R_temp_mean = None
        self.R_temp_stdev = None
        self.R_temp_max = None
        self.R_temp_min = None
        self.C_start_temp = None
        self.C_end_temp = None
        # OCV & mAh info:
        self.C_start_OCV = None
        self.C_start_mAh = None
        self.R_end_OCV = None
        self.R_end_mAh = None
        # Resistance info:
        self.C_start_R0 = None
        self.C_start_R0_dt = None
        self.C_end_R0 = None
        self.C_end_R0_dt = None

class PulseSequence:
    def __init__(self):
    #Pulse_list will be a list of Pulse objects    
        self.start_ind = None
        self.end_ind = None
        self.Pulse_list = []

class Acoustic_Pulse:
    def __init__(self):
        """
        Contains information for single pulses.
        """
        # Index labelling:
        self.R_pre_start_ind = None
        self.R_pre_end_ind = None
        self.C_start_ind = None
        self.C_end_ind = None
        self.R_post_start_ind = None
        self.R_post_end_ind = None
        # Temperature info:
        # During pulse
        self.C_temp_mean = None
        self.C_temp_stdev = None
        self.C_temp_max = None
        self.C_temp_min = None
        self.C_start_temp = None
        self.C_end_temp = None
        # Relaxation post pulse
        self.R_post_temp_mean = None
        self.R_post_temp_stdev = None
        self.R_post_temp_max = None
        self.R_post_temp_min = None
        self.R_post_start_temp = None
        self.R_post_end_temp = None
        # Relaxation pre pulse
        self.R_pre_temp_mean = None
        # OCV & mAh info:
        self.R_pre_end_OCV = None
        self.R_pre_end_mAh = None
        self.C_start_OCV = None
        self.C_start_mAh = None
        self.R_post_end_OCV = None
        self.R_post_end_mAh = None

class Acoustic_PulseSequence:
    def __init__(self):
    #Pulse_list will be a list of Pulse objects    
        self.start_ind = None
        self.end_ind = None
        self.Acoustic_Pulse_list = []

class EIS_object:
    def __init__(self):
        self.eis_df = None
        self.eis_id = None
        self.eis_start_datetime = None
        self.previous_ind = None # Latest index from df_cycling
        # For studies doing EIS repetitions:
        self.previous_step = None

class EIS_sequence:
    def __init__(self):
        self.EIS_list = []

class Acoustic_p2C:
    def __init__(self):
        self.start_ind = None
        self.end_ind = None
        self.ignore_ind_list = None

class Protocol_custom_objects:
    def __init__(self, test_id = None, cell_Q = None, P_char_chrg_seq = None,
        P_char_dischrg_seq = None, EIS_char_chrg_seq = None, EIS_char_dischrg_seq = None,
        EIS_other_seq = None, Acoustic_char_chrg_seq = None, Acoustic_char_dischrg_seq = None):
        #
        self.test_id = test_id
        self.cell_Q = cell_Q
        self.P_char_chrg_seq = P_char_chrg_seq
        self.P_char_dischrg_seq = P_char_dischrg_seq
        self.EIS_char_chrg_seq = EIS_char_chrg_seq
        self.EIS_char_dischrg_seq = EIS_char_dischrg_seq
        self.EIS_other_seq = EIS_other_seq
        self.Acoustic_char_chrg_seq = Acoustic_char_chrg_seq
        self.Acoustic_char_dischrg_seq = Acoustic_char_dischrg_seq

def json_to_custom_object(iterable):
    def get_attributes(custom_object):
        attributes = dir(custom_object)
        selected_attributes = []
        for item in attributes:
            # Ensure they are not default pyhton attributes
            if (item[:2] != '__') & (item[-2:] != '__'):
                selected_attributes.append(item)
        return(selected_attributes)

    custom_object = Protocol_custom_objects()
    attributes = get_attributes(custom_object)
    for attr in attributes:
        if (attr == 'Acoustic_char_chrg_seq') | (attr == 'Acoustic_char_dischrg_seq'):
            setattr(custom_object, attr, Acoustic_PulseSequence())
            pulse_list = []
            n_pulses = len(iterable[attr]['Acoustic_Pulse_list'])
            for i in range(n_pulses):
                pulse_dict = iterable[attr]['Acoustic_Pulse_list'][i]
                new_obj = Acoustic_Pulse()
                for key, value in pulse_dict.items():
                    setattr(new_obj, key, value)
                pulse_list.append(new_obj)
            
            getattr(custom_object, attr).Acoustic_Pulse_list = pulse_list
            for attr_2, value in iterable[attr].items():
                if attr_2 != 'Acoustic_Pulse_list':
                    setattr(
                        getattr(custom_object, attr), attr_2, value)
        elif (attr == 'P_char_chrg_seq') | (attr == 'P_char_dischrg_seq'):
            setattr(custom_object, attr, PulseSequence())
            pulse_list = []
            n_pulses = len(iterable[attr]['Pulse_list'])
            for i in range(n_pulses):
                pulse_dict = iterable[attr]['Pulse_list'][i]
                new_obj = Pulse()
                for key, value in pulse_dict.items():
                    setattr(new_obj, key, value)
                pulse_list.append(new_obj)
            getattr(custom_object, attr).Pulse_list = pulse_list
            for attr_2, value in iterable[attr].items():
                if attr_2 != 'Pulse_list':
                    setattr(
                        getattr(custom_object, attr), attr_2, value)
        elif (attr == 'EIS_other_seq'):
            setattr(custom_object, attr, EIS_sequence())
            eis_list = []
            n_pulses = len(iterable[attr]['EIS_list'])
            for i in range(n_pulses):
                eis_dict = iterable[attr]['EIS_list'][i]
                eis_df_dict = eis_dict['eis_df']
                new_obj = EIS_object()
                new_obj.eis_df = pd.DataFrame(eis_df_dict)
                for key, value in eis_dict.items():
                    if key != 'eis_df':
                        setattr(new_obj, key, value)               
                eis_list.append(new_obj)
            getattr(custom_object, attr).EIS_list = eis_list
            for attr_2, value in iterable[attr].items():
                if attr_2 != 'EIS_list':
                    setattr(
                        getattr(custom_object, attr), attr_2, value)
        # Special case for 'Acoustic_p2C'
        if 'Acoustic_p2C' in iterable.keys():
            custom_object.Acoustic_p2C = Acoustic_p2C()
            custom_object.Acoustic_p2C.start_ind = iterable['Acoustic_p2C']['start_ind']
            custom_object.Acoustic_p2C.end_ind = iterable['Acoustic_p2C']['end_ind']
            custom_object.Acoustic_p2C.ignore_ind_list = iterable['Acoustic_p2C']['ignore_ind_list']

    return (custom_object)

def create_custom_object(test_id, test_dir):
    # Protocol object
    file_name = '{}_Protocol_objects.json'.format(test_id)
    file_path = os.path.join(test_dir, file_name)
    with open(file_path, 'r') as f:
        json_string = f.read()
    Protocol_objects_json = json.loads(json_string)
    custom_object = json_to_custom_object(Protocol_objects_json)
    return custom_object

def smooth_by_convolution(s, window_len=11, kernel_type='rectangular', passes=10):
    """
    - s is a single signal.
    It is prepared by introducing reflected copies of the signal in both ends so that
    transient parts are minimized in the begining and end part of the output signal.
    Those copies have length window_len.
    """
    assert(window_len%2==1) # !!! Window_len must be an odd number.
    def extend_signal(s):
        """
        mirrored parts of the signal are introduced at the beginning and end of it.
        the mirroring is wrt both axes.
        """
        extra_front_bit = 2*s[0] - s[window_len:0:-1]
        extra_tail_bit = 2*s[-1] - s[-2:-window_len-2:-1]
        s_extended = np.r_[extra_front_bit, s, extra_tail_bit]
        return(s_extended)
    # ---------------------------------------
    s_extended = extend_signal(s)
    if kernel_type=='rectangular':
        # Equivalent to a moving average
        kernel = np.ones(window_len)
    elif kernel_type=='triangular':
        kernel = np.concatenate( (np.arange(1, window_len/2+1), np.arange(1, window_len/2)[::-1]) )
    elif kernel_type=='gaussian':
        x = np.arange(-(window_len+1)/2+1, (window_len+1)/2,1)
        kernel = 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2.)
    kernel = kernel/np.sum(kernel)
    for _ in range(passes):
        s_smooth = np.convolve(s_extended, kernel ,mode='same')
        s_smooth = s_smooth[window_len:-window_len]
        s_extended = extend_signal(s_smooth)
    return(s, s_smooth)

from scipy.signal import find_peaks
def signal_peaks(signals, crop_ind=0, window_len=15, passes=5, kernel_type='triangular',
    n_selected_peaks = 9):
    ###
    peak_indices = np.zeros(shape=(len(signals), n_selected_peaks))
    peak_heights = np.zeros(shape=(len(signals), n_selected_peaks))
    progress_iterator = iter(np.arange(10, 110, 10))
    pct_progress = 0
    print('Smoothing signals. Passes = {}. Window_len = {}'.format(passes, window_len))
    print('-----------------')
    for i, s in enumerate(signals):
        if i/len(signals)*100 > pct_progress:
            print('{pct:.2f} %'.format(pct = pct_progress))
            pct_progress = next(progress_iterator)
        s, s_smooth = smooth_by_convolution(
            s, window_len=window_len, kernel_type=kernel_type, passes=passes)
        #
        indices, _ = find_peaks(s_smooth[crop_ind:])
        indices += crop_ind
        heights = s[indices]
        #
        first_few_heights = heights[:n_selected_peaks-1]
        first_few_indices = indices[:n_selected_peaks-1]
        echo_peak_i = 2*round(len(heights)/3) + np.argmax(heights[2*round(len(heights)/3):])
        echo_peak_height = heights[echo_peak_i]
        echo_peak_index = indices[echo_peak_i]
        if echo_peak_index > first_few_indices[-1]:
            echo_peak_height = np.array([echo_peak_height])
            echo_peak_index = np.array([echo_peak_index])
        else:
            echo_peak_height = np.array([np.nan])
            echo_peak_index = np.array([np.nan])
        peak_heights[i] = np.concatenate((first_few_heights, echo_peak_height))
        peak_indices[i] = np.concatenate((first_few_indices, echo_peak_index))
    index_to_tof_ratio = 10/len(s_smooth)
    peak_tofs = peak_indices * index_to_tof_ratio
    return peak_heights, peak_tofs

def df_with_peaks(
    data_path, test_id, slicing_indices = [], ignore_ind_list = [],
    n_peaks = 9, **args, 
    ):
    import pandas as pd
    pk_headings = list(range(0, n_peaks))
    pk_headings = list(map(str, pk_headings))

    parquet_filename = '{}_acoustics_and_cycling.parquet'.format(test_id)
    parquet_filepath = os.path.join(data_path, test_id, parquet_filename)
    final_df = pq.read_table(parquet_filepath).to_pandas()
    
    if slicing_indices != []:
        start = slicing_indices[0]
        stop = slicing_indices[1]+1
        filter = final_df.index.isin(range(start, stop))
        final_df = final_df.loc[filter]
    if ignore_ind_list != []:
        filter = ~final_df.index.isin(ignore_ind_list)
        final_df = final_df.loc[filter]
    final_df = final_df.reset_index(drop=True)
    
    # Signal Processing
    signals = final_df['acoustics'].to_numpy()
    signal_len = signals.shape[1]
    crop_ind = int(
        signal_len/5) - np.argmax(np.flip(signals[:,:int(signal_len/5)],axis=1),
        axis=1).max()
    peak_heights, peak_tofs = signal_peaks(
        signals, crop_ind=crop_ind, n_selected_peaks=n_peaks, **args)

    pk_heights = pd.DataFrame(data = peak_heights, columns = pk_headings)
    pk_tofs = pd.DataFrame(data = peak_tofs, columns = pk_headings)

    df = pd.concat(
        [final_df['cycling'], final_df['acoustics'], pk_heights, pk_tofs], axis = 1,
        keys = ['cycling', 'acoustics', 'peak_heights', 'peak_tofs'])
    
    return df

