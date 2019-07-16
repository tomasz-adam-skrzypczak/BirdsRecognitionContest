from librosa import power_to_db
from librosa.feature import melspectrogram
import numpy as np

class LogMelSpectrogram:
    
    def __init__(self, params):
        
        self.sampling_rate = params["sr"]
        self.min_freq = params["min_freq"]
        self.max_freq = params["max_freq"]
        self.mels_no = params["mels"]
        
    def __call__(self, raw):
        
        melspec = melspectrogram(y = raw, sr = self.sampling_rate,
                                n_mels = self.mels_no, 
                                fmin = self.min_freq, fmax = self.max_freq)
        
        melspec = power_to_db(melspec, ref=np.max)
        melspec = melspec.reshape(1, melspec.shape[0], melspec.shape[1])
        
        return melspec
        