import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    '''
        inputs: [N, T, C]
        outputs: [N, T, C]
    '''
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        H = self.linear1(inputs)
        H = F.relu(H)
        T = self.linear2(inputs)
        T = torch.sigmoid(T)

        out = H * T + inputs * (1.0 - T)

        return out