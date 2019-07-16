from torch import device, softmax, Tensor
from torch.cuda import is_available
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm, trange

from EarlyStopping import EarlyStopping
import numpy as np


class Trainer():
    
    def __init__(self, training_time, path_to_models, weights = None):      

        self.epoches = training_time
        self.early_stop = EarlyStopping(path_to_models)
        self.device = device("cuda:0" if is_available() else "cpu")
        
        if weights is not None:
            class_weight = Tensor(weights).float().to(self.device)
            self.cost_function = CrossEntropyLoss(weight=class_weight)
        else:
            self.cost_function = CrossEntropyLoss()


    def AUC_ROC_metrics(self, model, data_loader):
        
        model_probabilities = []
        y_trues = []
        
        model.eval()

        for x, y in data_loader:
            x = x.to(self.device)
            
            y_hat = softmax(model(x), dim=1).to('cpu').detach().numpy()[:, 1]
            model_probabilities.append(y_hat)
            y_trues.append(y.numpy())
            
        model_probabilities = np.concatenate(model_probabilities).ravel()
        y_trues = np.concatenate(y_trues).ravel()
        
        return roc_auc_score(y_trues, model_probabilities)
    
    def train(self, model, optimizer, train_loader, test_loader):
        
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

        model = model.to(self.device)
        train_loss, scores, train_scores = [], [], []
        
        for i in trange(self.epoches):
            
            model.train()
            scheduler.step()
            
            for x, y in train_loader:
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                
                y_hat = model(x)

                loss = self.cost_function(y_hat, y.view((-1)))
                loss.backward()
                optimizer.step()
            
            train_loss.append(loss.item())
            
            train_auc_roc = self.AUC_ROC_metrics(model, train_loader)
            test_auc_roc = self.AUC_ROC_metrics(model, test_loader)
            scores.append(test_auc_roc)
            train_scores.append(train_auc_roc)
            
            isStop = self.early_stop(test_auc_roc, model)
            
        return train_loss, scores, train_scores
            