import numpy as np

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