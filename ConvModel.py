import torch
from torch.nn import Module, Conv2d, Conv1d, MaxPool2d, Sequential, AvgPool2d, ReLU, BatchNorm1d, BatchNorm2d, Linear, MaxPool1d, Dropout, AvgPool1d, GRU, LeakyReLU

class ConvBlock(Module):
    
    def __init__(self, in_dim, filters, kernel_size = 3, pad = 0):
        super(ConvBlock, self).__init__()
        
        self.layer = Sequential(Conv2d(in_dim, filters, kernel_size, padding = pad),
                                BatchNorm2d(filters),
                                ReLU())
    def forward(self, x):
        return self.layer(x)

class FlattenLayer(Module):
    
    def forward(self, x):
        return x.view(x.shape[0], -1)
    

class StackedFilterLayer(Module):
    def __init__(self, in_dim, no_layers, no_filters):
        super(StackedFilterLayer, self).__init__()
        
        layers = []
        for i in range(no_layers):
            layers.append(ConvBlock(in_dim, no_filters, kernel_size = 3, pad = 1))
            in_dim = no_filters
            
        self.layer = Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

    
class ResBlock(Module):
    def __init__(self, in_dim, no_layers, no_filters):
        super(ResBlock, self).__init__()
        
        layers = []
        res_layers = []

        if in_dim != no_filters:
            self.change_shape = Conv2d(in_dim, no_filters, kernel_size = 1)
        else:
            self.change_shape = lambda x : x
        
        for i in range(no_layers - 1):
            
            layers.append(Conv2d(in_dim, no_filters, kernel_size = 3, padding = 1))
            layers.append(BatchNorm2d(no_filters))
            layers.append(ReLU())
            in_dim = no_filters


        layers.append(Conv2d(in_dim, no_filters, kernel_size = 3, padding = 1))
        
        res_layers.append(BatchNorm2d(no_filters))
        res_layers.append(ReLU())
        
        self.conv_layers = Sequential(*layers)
        self.res_layers = Sequential(*res_layers)
        
        
    def forward(self, x):
        out = self.conv_layers(x)
        return self.res_layers(out + self.change_shape(x))
        

class CNNModel(Module):

    def __init__(self, in_dim, layers_size, filter_per_layer, linear_dim = None):
        super(CNNModel, self).__init__()
        
        self.args = [in_dim, layers_size, filter_per_layer, linear_dim]
        
        layers = []
        layers.append(BatchNorm2d(1))
        for i in range(len(layers_size)):
            layers.append(ResBlock(in_dim, layers_size[i], filter_per_layer[i]))
            layers.append(MaxPool2d(2))
            in_dim = filter_per_layer[i]
        
        layers.append(FlattenLayer())
        layers.append(Dropout())
        layers.append(Linear(linear_dim[0], linear_dim[1]))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)    
