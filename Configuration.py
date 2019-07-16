class Configuration:
    

    def __init__(self, params_dict):

        #write asserts later

        self.mels = params_dict['mels']
        self.min_frq = params_dict['min_freq']
        self.max_frq = params_dict['max_freq']
        self.sampling_rate = params_dict['sr']
        