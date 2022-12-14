import torch
import torch.nn as nn
import torch.nn.functional as F

from Conv1d import Conv1d
from Conv1dBank import Conv1dBank
from BatchNorm1d import BatchNorm1d
from MaxPool1d import MaxPool1d
from Highway import Highway

from Hyperparameters import Hyperparameters as hp

class CBHG(nn.Module):
    '''
    input:
        prenet_output: [N, T, E//2]
    output:
        outputs [N, T, E]
    '''
    def __init__(self):
        super().__init__()

        self.conv1d_bank = nn.Sequential(
            Conv1dBank(K=hp.K, in_channels=hp.E // 2, out_channels=hp.E // 2), # [N, T, E//2 * K],
            MaxPool1d(kernel_size=2) # [N, T, E//2 * K]
        )

        self.projection = nn.Sequential(
            Conv1d(in_channels=hp.K * hp.E // 2, out_channels=hp.E // 2, kernel_size=3),  # [N, T, E//2]
            BatchNorm1d(num_features=hp.E // 2),
            nn.ReLU(),
            Conv1d(in_channels=hp.E // 2, out_channels=hp.E // 2, kernel_size=3), # [N, T, E//2]
            BatchNorm1d(num_features=hp.E // 2)
        )

        self.highways = nn.Sequential()
        for i in range(hp.num_highways):
            self.highways.append(Highway(in_features=hp.E // 2, out_features=hp.E // 2))
            self.highways.append(nn.ReLU())

        self.gru = nn.GRU(input_size=hp.E // 2, hidden_size=hp.E // 2, num_layers=2, bidirectional=True, batch_first=True)
    
    def forward(self, inputs, prev_hidden=None): # input from prenet
        
        outputs = self.conv1d_bank(inputs)

        outputs = self.projection(outputs)

        outputs = outputs + inputs # residual

        # highway
        self.highways(outputs)
        
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)  # outputs [N, T, E]

        return outputs, hidden

if __name__ == "__main__":
    
    from torchsummaryX import summary

    model = CBHG().to(hp.device)
    inputs = torch.randn(64, 159, 128).to(hp.device)
    out, hidden = model(inputs)
    print(out.shape, hidden.shape)

    # summary(model, inputs)
