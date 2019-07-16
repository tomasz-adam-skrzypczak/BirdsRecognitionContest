import numpy as np
import hashlib
import torch
import datetime
import os

class EarlyStopping:

    def __init__(self, model_paths, patience = 10):
        
        self.best_score = np.NINF
        self.patience = patience
        self.saving_models_path = model_paths
        self.iteration = 0
    
    def __call__(self, score, model):
        
        if self.best_score < score:
            self.iteration = 0
            self.best_score = score
            self.save(model)
            
        else:
            self.iteration += 1
        
        if self.iteration == self.patience:
            return True
        else:
            return False
    
    def save(self, model):
        
        if self.best_score > 0.87:
            model_name = hashlib.md5(str(datetime.datetime.now()).encode('utf-8')).hexdigest() + '.pth'
            paths = os.path.join(self.saving_models_path, model_name)

            parameters = model.args
            
            model_dict = { "args" : parameters, "points" : self.best_score, "state_dict" : model.state_dict()}
            torch.save(model_dict, paths)
            