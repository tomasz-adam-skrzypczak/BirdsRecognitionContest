import numpy as np
from librosa.effects import hpss

class Adding_White_Noise:

    def __call__(self, signal):
        
        noise_amp = 0.005*np.random.uniform()*np.amax(signal)
        new_signal = signal + noise_amp * np.random.normal(size=signal.shape[0])
        
        return new_signal
    
class Amps_Augumentation:
    
    def __init__(self):
        self.power_generator = np.random.uniform(low=1.5, high=3.)

    def __call__(self, signal):        
        new_data = signal * self.power_generator
            
        return new_data 
    
class Hpps_Augumentation:
    
    def __call__(self, signal):
        
        harmonic_com, percussive_com = hpss(signal)
        new_data = percussive_com

        return percussive_com

class Pitch_Shifting:
    
    def __init__(self, pitch_pm = 2, bins_per_octave = 15):
        self.bins_per_octave = bins_per_octave
        self.pitch_pm = pitch_pm
        self.sampling_rate = 44100
    
    def __call__(self, signal):

        pitch_change =  self.pitch_pm * 2 * (np.random.uniform())   
        
        sh_signal = pitch_shift(signal.astype('float64'), 44100, n_steps=pitch_change, 
                        bins_per_octave=self.bins_per_octave)
        
        return sh_signal
    