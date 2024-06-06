import numpy as np
import os
import json
import pandas as pd
import pyarrow.parquet as pq
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import tensorflow as tf

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
        if ((attr == 'Acoustic_char_chrg_seq') | (attr == 'Acoustic_char_dischrg_seq')) & \
            (attr in str(iterable)):
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
        elif ((attr == 'P_char_chrg_seq') | (attr == 'P_char_dischrg_seq')) & \
            (attr in str(iterable)):
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
        elif (attr == 'EIS_other_seq') & (attr in str(iterable)):
            if iterable[attr] != None:
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

def save_figure(fig, visualisation_path, save_filename, format='pdf'):
    """
    Type can be pdf. If any other value is specified a png will be saved.
    """
    if not os.path.exists(visualisation_path):
        os.makedirs(visualisation_path)
    if format!='pdf':
        format='png'
        save_filename += '.png'
    else:
        save_filename += '.pdf'
    fig.savefig(os.path.join(visualisation_path, save_filename), bbox_inches='tight', format=format)

def make_spectrogram(waveform, frame_length=501, frame_step=5,
                     pad_end=False, crop_freq = 21):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=frame_length, frame_step=frame_step, pad_end=pad_end)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    # print(spectrogram.shape)
    # print('N features = {}'.format(spectrogram.shape[0]*spectrogram.shape[1]))
    if crop_freq != None:
        spectrogram = spectrogram[:, 1:crop_freq]
    else:
        spectrogram = spectrogram[:, 1:]
    return spectrogram

def spectrogram_data_for_plot(spectrogram):
    # If needed, remove the last axis which is for ML
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    return log_spec

def animate_signals(df_cycling, signals, fft_magns=False, freqs_MHz = False,
    drc=False, peak_heights=False, peak_tofs=False, crop_ind=False, temp_lims=(18,32),
    figsize = (12,4.8), dpi=150, annot_pos=(None, None), title=None,
    save_dir=None, save_name=None, time_step=2.5e-03, fps=30):
    """
    https://jckantor.github.io/CBE30338/A.03-Animation-in-Jupyter-Notebooks.html
    """
    # Some data and exception handling
    #---------------------------------
    if  (type(fft_magns) != bool) & (type(freqs_MHz) == bool):
        raise ValueError('Provide_frequencies in MHz to accompany the fft_mangs input')
    voltages = df_cycling["V(V)"].to_numpy()
    temperatures = df_cycling["Temp(degC)"].to_numpy()
    if drc==False:
        rates = df_cycling["C-Rate"].to_numpy()
        if 'CV' in df_cycling['Regime']:
            cv_filter = df_cycling['Regime']=='CV'
            rates[cv_filter]='CV'
        rate_cols = {0: 'r', 0.05: 'y', 0.2: 'b', 0.5: 'g', 0.75: 'k', 1.0: 'm', 'CV': 'c'}
    anim_indices = df_cycling.index.to_numpy()

    # Create the background frame
    #----------------------------
    if (type(fft_magns) == bool):
        f, axs = plt.subplots(1,3, gridspec_kw={'width_ratios': [8, 0.25, 0.25]},
            constrained_layout = True, figsize = figsize)
        if annot_pos == (None, None):
            annot_pos = (0.7, 0.9)       
    else:
        fft_magns /= 1000 # To make the axes plotting it look nicer.
        f, axs = plt.subplots(1,4, gridspec_kw={'width_ratios': [4, 0.25, 0.25, 4]},
            constrained_layout = True, figsize = figsize)
        line_freq_domain, = axs[3].plot([], [], marker='o', markersize=5)
        if annot_pos == (None, None):
            annot_pos = (0.6, 0.9)
    f.patch.set_facecolor('white')
    #
    if crop_ind!= False:
        line1_time_domain, = axs[0].plot([], [], c = 'tab:blue')
    line2_time_domain, = axs[0].plot([], [], c = 'tab:orange')
    barV, = axs[1].bar("V", height = 0)
    barT, = axs[2].bar("T", height = 0)
    if type(peak_heights) != bool:
        n_peaks = peak_heights.shape[1]
        peaks_colorscheme = cm.Greens(np.linspace(0.2, 1, n_peaks))
        scat_peaks = axs[0].scatter(
            np.arange(n_peaks), np.arange(n_peaks), marker='o', s = 90,
            c = peaks_colorscheme, edgecolors='tab:green')   
    my_text = axs[0].text(annot_pos[0], annot_pos[1], '',
        transform=axs[0].transAxes, backgroundcolor = 'white')    
    title_font_size = 18
    # axs[0] (line: time domain)
    time_domain_value_range = signals[anim_indices].max() - signals[anim_indices].min()
    offset = 0.1 * time_domain_value_range
    axs[0].set_ylim(signals[anim_indices].min() - offset,       
        signals[anim_indices].max() + offset)
    axs[0].set_xlim(0, 10)
    axs[0].set_xlabel('ToF (μs)')
    axs[0].set_ylabel('Acoustic Amplitude (a.u.)')
    # axs[1] (bar)
    axs[1].set_ylim(2.5,4.4)
    # axs[2] (bar)
    axs[2].set_ylim(temp_lims)
    # axs[3] (line: freq domain)
    if type(fft_magns) != bool:
        freq_domain_value_range = fft_magns[anim_indices, 1:].max() - fft_magns[anim_indices, 1:].min()
        offset = 0.1 * freq_domain_value_range
        axs[3].set_ylim(-0.01,
            fft_magns[anim_indices, 1:].max() + offset)
        offset = fft_magns.shape[1]*0.05
        axs[3].set_xlim(-offset, fft_magns.shape[1]+ offset)
        axs[3].set_xlim(freqs_MHz[0], freqs_MHz[-1])
        axs[3].set_xlabel('MHz')
        axs[3].set_ylabel('FFT magn. x 1000\n(a.u.)')
    # General
    axs[0].grid()
    f.suptitle(title, fontsize=title_font_size)
    f.align_ylabels()

    # TO help produce a progress update
    progress_iterator = iter(np.arange(0.5, 100.5, 0.5))
    global pct_progress
    pct_progress = 0

    # animation function. This is called sequentially
    # -----------------------------------------------
    def drawframe(n):
        # Print progress information
        if n/len(anim_indices)*100 > globals()["pct_progress"]:
            print('{pct:.2f} %'.format(pct = pct_progress))
            globals()["pct_progress"] = next(progress_iterator)

        x_time_domain = np.arange(len(signals[n])) * time_step + time_step # The first datapoint logged is at time_step, not zero.
        if crop_ind!= False:
            x1_time_domain = x_time_domain[:crop_ind].tolist()
            y1_time_domain = signals[n, :crop_ind].tolist()
            x2_time_domain = x_time_domain[crop_ind:].tolist()
            y2_time_domain = signals[n, crop_ind:].tolist()
            line1_time_domain.set_data(x1_time_domain, y1_time_domain)
        else:
            x2_time_domain = x_time_domain.tolist()
            y2_time_domain = signals[n].tolist()
        line2_time_domain.set_data(x2_time_domain, y2_time_domain)
        ## axs[1] (bar)
        volts = voltages[n]
        barV.set_height(volts)
        ## axs[2] (bar)
        temp = temperatures[n]
        barT.set_height(temp)
        ## ax4 (line: freq domain)
        if type(fft_magns) != bool:
            fft_magn = fft_magns[n]
            line_freq_domain.set_data(np.arange(len(fft_magn)), fft_magn)
            line_freq_domain.set_data(freqs_MHz[1:], fft_magn[1:])
        ## Annotations
        if drc==False:
            rate = rates[n]
            text_col = rate_cols[rate]
            plt.setp(my_text, c = text_col)
            if rate != 'CV':
                rate_annotation = 'Signal Id: %s \nRate: %s C'%(n, rate)
            else:
                rate_annotation = 'Signal Id: %s \nRate: %s'%(n, rate)
            my_text.set_text(rate_annotation)    
        else:
            plt.setp(my_text, c = 'k')
            my_text.set_text('Signal Id: %s'%(n))
        if type(peak_heights) != bool:
            x_scat = peak_tofs[n]
            y_scat = peak_heights[n]
            scat_array = np.concatenate((x_scat.reshape(len(x_scat),1),y_scat.reshape(len(y_scat),1)), axis=1)
            scat_peaks.set_offsets(scat_array)
        return line2_time_domain, barV, barT,

    import matplotlib.animation as animation
    # interval is the delay between frames in milliseconds.
    ani = animation.FuncAnimation(f, drawframe, interval = 20,
        blit=True, save_count=len(anim_indices))
    if save_name != None:
        ani.save(os.path.join(save_dir, save_name)+'.mp4', dpi = dpi, fps=fps)
    else:
        return ani

def animate_spectrograms(df_cycling, signals, specs, save_dir=None, save_name=None,
                        dpi=150, fps=30, temp_lims=(18,32),
                        frame_length=501, frame_step=5, crop_freq = 21,
                        title = None):
    
    """"
    specs means spectrograms (a 3D numpy array)
    """
    time_step=2.5e-03
    freqs = np.fft.rfftfreq(n=frame_length, d = time_step)[1:crop_freq]
    voltages = df_cycling["V(V)"].to_numpy()
    temperatures = df_cycling["Temp(degC)"].to_numpy()
    rates = df_cycling["C-Rate"].to_numpy()
    if 'CV' in df_cycling['Regime']:
        cv_filter = df_cycling['Regime']=='CV'
        rates[cv_filter]='CV'
    rate_cols = {0: 'r', 0.05: 'y', 0.2: 'b', 0.5: 'g', 0.75: 'k', 1.0: 'm', 'CV': 'c'}
    annot_pos = (0.7, 0.9) 

    # Create the background frame
    f, axs = plt.subplots(1,4, figsize=(12, 4.8),
            gridspec_kw={'width_ratios': [4, 0.25, 0.25, 4]}, constrained_layout = True)
    f.patch.set_facecolor('white')
    # Line
    line_time_domain, = axs[0].plot([], [], c = 'tab:orange')
    # Bars
    barV, = axs[1].bar("V", height = 0)
    barT, = axs[2].bar("T", height = 0)
    # Spectrogram
    specs_log_max = np.log(specs.max())
    specs_log_min = np.log(specs.min())

    init_spectrogram = specs[0]
    log_spec = spectrogram_data_for_plot(init_spectrogram)
    # Plot the x values in the middle of the time window they represent.
    X = (758*time_step + 
         (np.array(range(log_spec.shape[1]))*frame_step)*time_step + 
         round(frame_length/2)*time_step)
    freqs = np.fft.rfftfreq(n=frame_length, d = time_step)[1:crop_freq] #MHz
    pcm = axs[3].pcolormesh(X, freqs, log_spec)

    my_text = axs[0].text(annot_pos[0], annot_pos[1], '',
        transform=axs[0].transAxes, backgroundcolor = 'white') 

    # Axes limits
    axs[0].set_xlim(time_step*758, 10)
    axs[1].set_ylim(2.5,4.4)
    axs[2].set_ylim(temp_lims)
    axs[3].set_xlim(time_step*758, 10)
    time_domain_value_range = signals.max() - signals.min()
    offset = 0.1 * time_domain_value_range
    axs[0].set_ylim(signals.min() - offset,       
        signals.max() + offset)
    
    # Axes labels
    axs[0].set_xlabel('ToF (μs)')
    axs[0].set_ylabel('Acoustic Amplitude (a.u.)')
    axs[3].set_xlabel('ToF (μs)')
    axs[3].set_ylabel('Frequency (MHz)')

    # General
    axs[0].grid()
    title_font_size = 18
    f.suptitle(title, fontsize=title_font_size)

    # TO help produce a progress update
    progress_iterator = iter(np.arange(0.5, 100.5, 0.5))
    global pct_progress
    pct_progress = 0

    def drawframe(n):
        # Print progress information
        if n/len(df_cycling)*100 > globals()["pct_progress"]:
            print('{pct:.2f} %'.format(pct = pct_progress))
            globals()["pct_progress"] = next(progress_iterator)

        x_time_domain = np.arange(len(signals[n])) * time_step + time_step # The first datapoint logged is at time_step, not zero.
        y_time_domain = signals[n]
        line_time_domain.set_data(x_time_domain + time_step*758, y_time_domain)

        volts = voltages[n]
        barV.set_height(volts)

        temp = temperatures[n]
        barT.set_height(temp)

        spec = specs[n]
        log_spec = spectrogram_data_for_plot(spec)
        # pcm.set_data(X, Y, log_spec)
        pcm.set_array(log_spec.ravel())
        pcm.set_clim(vmin=np.min(log_spec), vmax=np.max(log_spec))
        # pcm.set_clim(vmin=specs_log_min, vmax=specs_log_max)

        rate = rates[n]
        text_col = rate_cols[rate]
        plt.setp(my_text, c = text_col)
        if rate != 'CV':
            rate_annotation = 'Signal Id: %s \nRate: %s C'%(n, rate)
        else:
            rate_annotation = 'Signal Id: %s \nRate: %s'%(n, rate)
        my_text.set_text(rate_annotation) 

        return line_time_domain, barV, barT, pcm,

    import matplotlib.animation as animation
    # interval is the delay between frames in milliseconds.
    ani = animation.FuncAnimation(f, drawframe, interval = 20,
        blit=True, save_count=len(df_cycling))
    if save_name != None:
        ani.save(os.path.join(save_dir, save_name)+'.mp4', dpi = dpi, fps=fps)
    else:
        return ani

#CCCV data plotting------------------------------------
#------------------------------------------------------
def colorscheme(n_increments, cmap='Blues'):
    return iter(
        mpl.colormaps[cmap](np.linspace(0.2, 1, n_increments))
    )

def multi_cell_plot(df, cells, cell_aliases, x_quantity = 'Q(mAh)', c_rates = [1], xlims = (0, 250), figsize=(15, 14.4), dpi=300,
    domain = 'time', relative_peaks = True, freqs_1d = None, freq_ind_pair = (),
    title_font_size = 18, subtitle_font_size = 16, axlabels_font_size = 14, ticksize=12, label_text_size=14,
    save_filename=None, visualisation_path=None, return_axes=False):
    """
    df must be a pandas DataFrame which is multiindex with:
    df.columns.levels[0] = Index(['cycling', 'acoustics', 'peak_heights', 'peak_tofs',
        'fft_magns', 'fft_freq_indices_sorted'], dtype='object')
    cell: A list of cell names as defined in the database.
    cell_aliases: A dictionary where the keys are the cell names and the values are cell_aliases
    - which will be used instead of the cell names used in the database.
    Options:
    domain: 'time' or 'freq'
        if 'time'; 
        - relative_peaks: Bool
        if 'freq'; must also provide an array 'freqs_id' of the frequencies in MHz
        - freq_ind_pair: provide the indices of two frequencies to plot.
    """
    if len(c_rates)==1:
        c_rate_for_title = 'C-rate = {} C.'.format(c_rates[0])
    elif len(c_rates)==2:
        c_rate_for_title  = 'C-rates = {}, {} C.'.format(c_rates[0], c_rates[1])
    elif len(c_rates)==3:
        c_rate_for_title  = 'C-rates = {}, {}, {} C.'.format(c_rates[0], c_rates[1], c_rates[2])
    #
    if domain=='time':
        n_peaks = df['peak_heights'].shape[1]
        if relative_peaks==True:
            n_rows = 7
            peak_2_tof_row = 4
            peak_last_tof_row = 5
            y_labels = [
                'Ampl. (a.u.)\n(second peak)',
                'Ampl. (a.u.)\n(last peak)',
                'Ampl. Diff.\n(second & last\npeaks)',
                'ToF (μs)\n(second peak)',
                'ToF (μs)\n(last peak)',
                'ToF Diff.\n(second & last\npeaks)',
            ]
        elif relative_peaks==False:
            n_rows = 5
            peak_2_tof_row = 3
            peak_last_tof_row = 4
            y_labels = [
                'Ampl. (a.u.)\n(second peak)',
                'Ampl. (a.u.)\n(last peak)',
                'ToF (μs)\n(second peak)',
                'ToF (μs)\n(last peak)',
            ]
        title = 'Time-domain peaks. {}\nSecond and last acoustic peaks.'.format(c_rate_for_title)

    elif domain=='freq':
        if freq_ind_pair != ():
            n_rows = 4
            y_labels = [
                'FFT magn. (a.u. x1000)\n({:.2f} MHz)'.format(freqs_1d[freq_ind_pair[0]]),
                'FFT magn. (a.u.x1000)\n({:.2f} MHz)'.format(freqs_1d[freq_ind_pair[1]]),
                'FFT magn. Diff between\nselected freq. components'
            ]
        title = 'Freq-domain. {}'.format(c_rate_for_title)
    #
    f, axs = plt.subplots(n_rows,7, sharex='col', sharey='row', figsize=figsize, constrained_layout=True, dpi=dpi)
    f.patch.set_facecolor('white')
    for j, c_rate in enumerate(c_rates):
        filter1 = df['cycling']['C-Rate'] == c_rate
        cycles = df['cycling'].loc[filter1, 'Cycle'].unique()
        if c_rate == 0.2:
            col_scheme = 'Purples'
        elif c_rate == 0.5:
            col_scheme = 'Oranges'
        elif c_rate ==1:
            col_scheme = 'Blues'
        for i in range(len(cells)):
            cell_id = cells[i]
            filter2 = df['cycling']['Cell_ID'] == cell_id
            color = colorscheme(len(cycles), col_scheme)
            for cycle in cycles:
                c = next(color)
                label = cycle+1 # To display cycles starting at 1 instead of 0
                filter3 = df['cycling']['Cycle'] == cycle
                filter = filter1 & filter2 & filter3
                x = df['cycling'].loc[filter, x_quantity].to_numpy()
                # For each C rate use a separate axs to produce labels.
                if j == i:
                    if j == 0:
                        axs[0,i].plot(
                            x,
                            df['cycling'].loc[filter, 'V(V)'], c=c, label=label,
                            )
                    elif j == 1:
                        axs[0,i].plot(
                            x,
                            df['cycling'].loc[filter, 'V(V)'], c=c, label=label,
                            )
                    elif j == 2:
                        axs[0,i].plot(
                            x,
                            df['cycling'].loc[filter, 'V(V)'], c=c, label=label,
                            )
                else:
                    axs[0,i].plot(
                        x,
                        df['cycling'].loc[filter, 'V(V)'], c=c,
                        )             
                if domain == 'time':
                    axs[1,i].plot(
                        x,
                        df['peak_heights'].loc[filter, '1'], c=c,
                    )
                    axs[2,i].plot(
                        x,
                        df['peak_heights'].loc[filter, str(n_peaks-1)], c=c,
                    )
                    if relative_peaks:
                        axs[3,i].plot(
                            x,
                            (df['peak_heights'].loc[filter, str(n_peaks-1)]-
                            df['peak_heights'].loc[filter, '1']), c=c,
                        )
                        axs[6,i].plot(
                            x,
                            (df['peak_tofs'].loc[filter, str(n_peaks-1)]-
                            df['peak_tofs'].loc[filter, '1']), c=c,
                        )
                    axs[peak_2_tof_row,i].plot(
                        x,
                        df['peak_tofs'].loc[filter, '1'], c=c,
                    )
                    axs[peak_last_tof_row,i].plot(
                        x,
                        df['peak_tofs'].loc[filter, str(n_peaks-1)], c=c,
                    )
                elif domain == 'freq':
                    if freq_ind_pair != ():
                        axs[1,i].plot(
                            x, df['fft_magns'].loc[filter, str(freq_ind_pair[0])]/1000, c=c,
                        )
                        axs[2,i].plot(
                            x, df['fft_magns'].loc[filter, str(freq_ind_pair[1])]/1000, c=c,
                        )
                        axs[3,i].plot(
                            x,
                            (df['fft_magns'].loc[filter, str(freq_ind_pair[0])]-
                            df['fft_magns'].loc[filter, str(freq_ind_pair[1])])/1000, c=c,
                        )
        #
        if j==0:
            handles, _ = axs[0,0].get_legend_handles_labels()
            f.legend(handles = handles, 
                loc='center left', bbox_to_anchor=(1, 0.8), ncol=2, title="Cycle ({}C)".format(c_rate), 
                fontsize=label_text_size, title_fontsize=label_text_size)
        elif j ==1:
            handles, _ = axs[0,1].get_legend_handles_labels()
            f.legend(handles = handles, 
                loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, title="Cycle ({}C)".format(c_rate), 
                fontsize=label_text_size, title_fontsize=label_text_size)
        elif j ==2:
            handles, _ = axs[0,2].get_legend_handles_labels()
            f.legend(handles = handles,
                loc='center left', bbox_to_anchor=(1, 0.2), ncol=2, title="Cycle ({}C)".format(c_rate), 
                fontsize=label_text_size, title_fontsize=label_text_size)
    #
    if 'SoC' in x_quantity:
        x_label = 'SoC'
    else:
        x_label = x_quantity
    axs[0,0].set_ylabel('V(V)', fontsize=axlabels_font_size)
    for i in range(n_rows-1):
        axs[i+1, 0].set_ylabel(y_labels[i], fontsize=axlabels_font_size)
    for i in range(len(cells)):
        cell_id = cells[i]
        axs[-1, i].set_xlabel(x_label, fontsize=axlabels_font_size)
        axs[-1, i].set_xlim(xlims)
        axs[0,i].set_title('Cell {}'.format(cell_aliases[cell_id]), fontsize=subtitle_font_size)
    
    mpl.rc('xtick', labelsize=ticksize)
    mpl.rc('ytick', labelsize=ticksize)
    f.align_ylabels(axs[:, 0])
    f.suptitle(title, fontsize=title_font_size)
    if save_filename != None:
        save_path = os.path.join(visualisation_path, save_filename)
        f.savefig(save_path, bbox_inches='tight')
    if return_axes:
        return(f, axs)

def plot_cycling_data(df_cycling, df_peak_tofs, f, axs):
    # Plot Cycling data only
    filter = df_cycling['Cycle'] != 999
    unique_cycles = df_cycling.loc[filter, 'Cycle'].unique()
    for cycle in unique_cycles:
        filter_cycle = df_cycling['Cycle']==cycle
        temp_df = df_cycling.loc[filter_cycle]
        time_h = temp_df['Time(s)'].to_numpy()/3600
        axs[0].plot(time_h, temp_df['C-Rate'], c='k')
        axs[1].plot(time_h, temp_df['V(V)'], c='tab:blue')
        axs[2].plot(time_h, temp_df['Temp(degC)'], c='tab:blue')
        axs[3].plot(time_h, temp_df['Q(mAh)'], c='tab:blue')
        filter_pol = temp_df['Polarisation'] == '-'
        capacity = temp_df.loc[filter_pol, 'Q(mAh)'].iloc[0] - temp_df.loc[filter_pol, 'Q(mAh)'].iloc[-1]
        capacity_time = temp_df.loc[filter_pol, 'Time(s)'].iloc[-1]/3600
        # Capacity_datetime = temp_df.loc[filter, 'Datetime'].iloc[-1]
        axs[4].scatter(capacity_time, capacity, c='tab:blue')
        # Back wall echo peak
        echo_peak = df_peak_tofs.loc[filter_cycle, '8'].to_numpy()
        axs[5].plot(time_h, echo_peak, c='tab:blue')

    # Draw a line at Q=0
    xlims = axs[3].get_xlim()
    axs[3].plot([xlims[0], xlims[1]], [0,0], c='k', linestyle=(0, (1, 10)))
    axs[3].set_xlim(xlims)
    # Define consistent y limits for the Temperature, Q and capacity axes
    axs[2].set_ylim(18, 27)
    axs[2].set_yticks([18, 21, 24, 27])
    axs[3].set_ylim(-10, 245)
    axs[3].set_yticks([0, 100, 200, 245])
    axs[4].set_ylim(125, 235)
    axs[4].set_yticks([125, 165, 200, 235])
    #
    axs[0].set_ylabel('C-rate')
    axs[1].set_ylabel('V (V)')
    axs[2].set_ylabel('Temp ($^\circ$C)')
    axs[3].set_ylabel('Q (mAh)')
    axs[4].set_ylabel('Capacity\n(mAh)')
    axs[5].set_ylabel('Back wall\necho peak\nToF (μs)')
    axs[0].set_yticks([0.2, 0.5, 1], ['C/5', 'C/2', '1C'])
    axs[-1].set_xlabel('Time (h)')
    f.align_ylabels()

