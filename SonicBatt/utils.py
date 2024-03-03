import numpy as np

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

class Protocol_custom_objects:
    def __init__(self, test_id = None, cell_Q = None, P_char_chrg_seq = None,
        P_char_dischrg_seq = None, EIS_char_chrg_seq = None, EIS_char_dischrg_seq = None,
        EIS_other_seq = None, Acoustic_char_chrg_seq = None, Acoustic_char_dischrg_seq = None):
        
        self.test_id = test_id
        self.cell_Q = cell_Q
        self.P_char_chrg_seq = P_char_chrg_seq
        self.P_char_dischrg_seq = P_char_dischrg_seq
        self.EIS_char_chrg_seq = EIS_char_chrg_seq
        self.EIS_char_dischrg_seq = EIS_char_dischrg_seq
        self.EIS_other_seq = EIS_other_seq
        self.Acoustic_char_chrg_seq = Acoustic_char_chrg_seq
        self.Acoustic_char_dischrg_seq = Acoustic_char_dischrg_seq

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


