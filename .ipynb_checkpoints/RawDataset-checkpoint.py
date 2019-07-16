import os

import librosa
import numpy as np

from sklearn.model_selection import train_test_split


class RawDataset:

    def __init__(self, train_path, test_path, representation_method):
        
        self.train_path = train_path
        self.test_path = test_path
        self.representation_method = representation_method
        
        self.train_x_raw = []
        self.train_x_rep = []
        self.train_y = []
        
        self.test_set = []
        self.load_train()
        self.load_test_dataset()
    
    def read_labels(self):

        labels = []
        with open(os.path.join(self.train_path, 'labels.txt'), 'r') as file:
            text = file.read()
            for line in text.split('\n')[1:]:
                if len(line) > 1:
                    rec, start, stop = line.split(',')
                    rec, start, stop = int(rec[3:]), float(start), float(stop)
                    labels.append([rec, start, stop])
                    
        return np.array(labels)


    def check_voices(self, second, labels, tol=0.):

        return (labels[1] >= second and labels[1] < second + 1 - tol) or \
               (labels[2] < second + 1 and labels[2] > second + tol) or \
               (labels[1] < second and labels[2] > second + 1)


    def map_seconds_to_y(self, labels):

        y = [0] * 10
        y_restrictive = [0] * 10
        for s in range(10):
            for l in labels:
                if self.check_voices(s, l):
                    y[s] = 1
                if self.check_voices(s, l, 0.02):
                    y_restrictive[s] = 1
            if y[s] != y_restrictive[s]:
                y[s] = -1
        return np.array(y)
    
    def load_raw(self, filename):
        raw, _ = librosa.core.load(filename, sr = None)
        return np.split(raw, 10)
    
    def load_train(self):
        
        filenames_no = len([f_name for f_name in os.listdir(self.train_path) if f_name.endswith(".wav")])        
        labels = self.read_labels()

        for file_idx in range(1, filenames_no + 1, 1):
            
            recording_labels = labels[labels[:, 0] == file_idx]
            y = self.map_seconds_to_y(recording_labels)
            
            filename = os.path.join(self.train_path, "rec" + str(file_idx) + ".wav")
            raw = np.array(self.load_raw(filename))
            mask = y != -1
            
            raw = raw[mask]
            y = y[mask]
            
            self.train_x_raw.append(raw)
            self.train_x_rep += [self.representation_method(r) for r in raw]
            self.train_y.append(y)
            
        self.train_x_raw = np.concatenate(self.train_x_raw, axis = 0)
        print(self.train_x_raw.shape)
        self.train_x_rep = np.stack(self.train_x_rep, axis = 0)
        print(self.train_x_rep.shape)
        self.train_y = np.concatenate(self.train_y, axis=0)
        
    def hierarchical_split(self, train_ratio):

        hierarchy = [48, 91, 103, 150, 167, 184, 202, 222, 241, 266, 297, 312, 326, 337, 350, 359]
        hierarchy = [x * 10 for x in hierarchy]

        X_train_raw, X_train_rep, X_valid = [], [], []
        y_train, y_valid = [], []
        
        test_ratio = 1. - train_ratio
        start = 0
        
        for end in hierarchy:
            x_tr_raw, _, x_tr_rep, x_val_rep, y_tr, y_val = train_test_split(
                                                      self.train_x_raw[start : end], self.train_x_rep[start : end],
                                                      self.train_y[start : end],
                                                      test_size = test_ratio, shuffle=True, 
                                                      stratify = self.train_y[start : end])
            start = end

            X_train_raw.append(x_tr_raw)
            X_train_rep.append(x_tr_rep)
            X_valid.append(x_val_rep)
            y_train.append(y_tr)
            y_valid.append(y_val)
            
        X_train_raw, X_train_rep, X_valid = np.concatenate(X_train_raw), np.concatenate(X_train_rep), np.concatenate(X_valid)
        y_train, y_valid = np.concatenate(y_train), np.concatenate(y_valid)

        return (X_train_raw, X_train_rep), y_train, X_valid, y_valid

    def wiser_split(self, train_ratio):
        #popraw pozniej
        test_ratio = 1. - train_ratio
        X_train, X_valid, y_train, y_valid = train_test_split(self.train_x, self.train_y, test_size = 0.2, shuffle=True, stratify=self.train_y)

        return X_train, y_train, X_valid, y_valid

    def get_train_val_set(self, mode = "hierarchical", train_ratio = 0.8):
        
        if mode == 'hierarchical':
            return self.hierarchical_split(train_ratio)
        if mode == 'wiser':
            return self.wiser_split(train_ratio)
        else:
            raise
            
    def load_test_dataset(self):
        
        with open('sampleSubmission.csv', 'r') as file:
            lines = file.read().split()[1:]
            sample_ids = [line.split(',')[0] for line in lines]
            samples = np.array([s.split('/') for s in sample_ids])

        X_test = []
        rec_files = sorted([file_name for file_name in os.listdir('test') 
                        if file_name.endswith('.wav')], key=lambda x: int(x.split('.')[0][3:]))
        
        for file_name in rec_files:
            recording_id = file_name.split('.')[0][3:]
            time_markers = samples[samples[:, 0] == recording_id, 1].astype(np.int)
            
            raws = self.load_raw(os.path.join(self.test_path, file_name))

            for t in time_markers:
                X_test.append(self.representation_method(raws[t]))
        
        self.test_set = np.array(X_test)


