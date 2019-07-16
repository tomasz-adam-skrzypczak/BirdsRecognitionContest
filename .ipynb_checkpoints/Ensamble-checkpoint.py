import torch
import os
import numpy as np

from ConvModel import CNNModel

class Ensamble:
    
    def __init__(self, models_path, no_model):
        self.models_path = models_path
        self.no_model = no_model
        self.best_models = []
        self.find_best_models()
        
    def find_best_models(self):
        
        models_filenames = [x for x in (os.listdir(self.models_path)) if not x.startswith('.')]
        scores = []
        
        for filename in models_filenames:
            m_path_full = os.path.join(self.models_path, filename)
            args = torch.load(m_path_full, map_location=torch.device("cpu"))
            scores.append([m_path_full, args['points']])
            
        scores = sorted(scores, reverse=True, key = lambda model : model[1])
        self.best_models = [mods[0] for mods in scores[:self.no_model]]
        print(self.best_models)
        
        
    def make_predictions(self, test_dl):
        
        predictions = []

        for model_path in self.best_models:
            args = torch.load(model_path, map_location=torch.device('cpu'))
            
            model = CNNModel(*args["args"])
            model.load_state_dict(args['state_dict'])
            model.eval()

            
            with torch.no_grad():
                ps = []

                for X in test_dl:
                    out = model(X[0])
                    ps.append(torch.softmax(out, dim = 1)[:, 1].detach().numpy())
                    
                predictions.append(np.concatenate(ps, axis = 0))
                
        return np.stack(predictions).mean(axis=0)