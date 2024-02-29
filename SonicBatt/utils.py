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