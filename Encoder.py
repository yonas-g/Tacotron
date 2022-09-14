import torch
import torch.nn as nn

from modules.PreNet import PreNet
from CBHG import CBHG

from Hyperparameters import Hyperparameters as hp

class Encoder(nn.Module):
    '''
    input:
        embedding: [N, T_x, E]
    output:
        outputs: [N, T_x, E]
        hidden: [2, N, E//2]
    '''
    def __init__(self):
        super().__init__()
        self.prenet = PreNet(in_features=hp.E)  # [N, T, E//2]
        self.cbhg = CBHG() # outputs [N, T, E]
    
    def forward(self, inputs, prev_hidden=None):
        
        inputs = self.prenet(inputs)
        outputs, hidden = self.cbhg(inputs, prev_hidden) # outputs [N, T, E]

        return outputs, hidden