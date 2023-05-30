import torch
import torch.nn as nn

def MLP():
    return nn.Sequential(
        nn.Linear(8,50),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(50,1))
