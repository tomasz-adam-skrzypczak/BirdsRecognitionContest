import librosa
import numpy as np

from torch.utils.data import Dataset
from torch import Tensor, LongTensor


class BirdsRecordingDataset(Dataset):
    
    def __init__(self, recordings, targets, representation_method, augumentation_methods, train = True, transform_probability = 0.5):
        
        if train is True:
            self.raw_recordings = recordings[0]
            self.rep_recordings = recordings[1]
        else:
            self.rep_recordings = recordings
            
        self.targets = targets
        
        self.representation_method = representation_method
        self.augumentation_methods = augumentation_methods
        self.transform_prob = transform_probability

    def __len__(self):
        return len(self.rep_recordings)
    
    def __getitem__(self, idx):
        
        x = self.rep_recordings[idx]
        y = self.targets[idx]
        
        if y == 1 and len(self.augumentation_methods) > 0 and np.random.rand() < self.transform_prob:

            aug_no = np.random.randint(len(self.augumentation_methods))
            transform = self.augumentation_methods[aug_no] 
            x = self.representation_method(transform(self.raw_recordings[idx]))

        return Tensor(x), y
