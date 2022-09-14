import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm1d(nn.Module):
    '''
    inputs: [N, T, C]
    outputs: [N, T, C]
    '''

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, inputs):
        out = self.bn(inputs.transpose(1, 2).contiguous())
        return out.transpose(1, 2)