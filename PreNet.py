import torch
import torch.nn as nn
from Hyperparameters import Hyperparameters as hp

class PreNet(nn.Module):
    '''
    inputs: [N, T, in]
    outputs:[N, T, E // 2]
    '''

    def __init__(self, in_features):
        super().__init__()
        
        self.prenet = nn.Sequential(
            nn.Linear(in_features, hp.E),
            nn.ReLU(),
            nn.Dropout(hp.dropout_p),
            nn.Linear(hp.E, hp.E // 2),
            nn.ReLU(),
            nn.Dropout(hp.dropout_p)
        )

    def forward(self, inputs):
        output = self.prenet(inputs)
        return output


if __name__ == "__main__":
    model = PreNet(hp.E)
    inputs = torch.autograd.Variable(torch.randn(10, 22, hp.E))
    out = model(inputs)
    print(out.shape)