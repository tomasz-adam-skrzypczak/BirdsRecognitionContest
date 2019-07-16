from torch.utils.data import DataLoader
from torch.cuda import is_available
from torch.nn import Sigmoid
from torch import device, softmax
from librosa.feature import mfcc, melspectrogram
from librosa import power_to_db
from librosa.decompose import nn_filter

import numpy as np

sigmoid = Sigmoid()

FILENAME_H = "recording_id"


def make_prediction(model, test_data):
    dev = device("cuda:0" if is_available() else "cpu")
    
    model = model.to(dev)
    dl = DataLoader(test_data, shuffle=False, batch_size=10)
    model.eval()
    probabilities = [softmax(model(x.to(dev)), dim=1)[:,1].view(-1).cpu().detach().numpy() for x in dl]
    
    return probabilities

def make_submission(model, test_data):
    probs = make_prediction(model, test_data)
    
    print(np.array(probs)[0, :])
    with open('sampleSubmission.csv', 'r') as file:
        lines = file.read().split()[1:]
        sample_ids = [line.split(',')[0] for line in lines]
        samples = np.array([s.split('/') for s in sample_ids])

    
    output_file = ['sample_id,prediction']
    
    for idx in range(len(probs)):
        times = samples[samples[:, 0] == str(idx + 1), 1].astype(np.int)
#         print(probs)
        sub_probs = probs[idx][times - 1]
    
        for i in range(len(times)):
            sample_output = str(idx + 1) + '/' + str(times[i]) + ',' + str(sub_probs[i])
            output_file.append(sample_output)
    
    with open('mySubmission.csv', 'w') as file:
        file.write('\n'.join(output_file) + '\n')
        
        
def getting_targets(filename, dt):
    audio_id = filename.split('/')[-1].split('.')[0]
    returned_value = dt.loc[dt[FILENAME_H] == audio_id]
    probs = np.zeros((10,))

    if not returned_value.empty:
        seconds = returned_value.values[:, [1]].astype(np.int).reshape((-1)) - 1
        probs[seconds] = 1

    return probs

def targets_to_shifting(filename, dt):
    audio_id = filename.split('/')[-1].split('.')[0]
    returned_value = dt.loc[dt[FILENAME_H] == audio_id]
    
    return returned_value[returned_value.columns[1:]].values
    

def check_voices(second, labels, tol=0.):

    return (labels[0] >= second and labels[0] < second + 1 - tol) or \
           (labels[1] < second + 1 and labels[1] > second + tol) or \
           (labels[0] < second and labels[1] > second + 1)


def map_seconds_to_y(labels):

    y = [0] * 10
    y_restrictive = [0] * 10
    for s in range(10):
        for l in labels:
            if check_voices(s, l):
                y[s] = 1
            if check_voices(s, l, 0.02):
                y_restrictive[s] = 1
        if y[s] != y_restrictive[s]:
            y[s] = -1
    return y

def make_labels_by_time(labels, time = 1.):
    no_labels = int(10 / time)
    y = no_labels * [0]
    coeff = no_labels / 10

    for l in labels:
        for idx in range(int(l[0] * coeff), int(l[1] * coeff) + 1, 1):
            y[idx] = 1
            
    return y
    
def make_melspec_(signal, config):

    spectogram = melspectrogram(signal, sr = config.sampling_rate, 
                               fmin = config.band_pass[0], fmax = config.band_pass[1], 
                               n_mels = config.n_mels)
#     spectogram = nn_filter(spectogram)
    spectogram = power_to_db(spectogram)
    if config.representation == 'shorter_melspec':
        return spectogram
    else:
        return spectogram.reshape((1, spectogram.shape[0], spectogram.shape[1]))
