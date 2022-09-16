import torch
import torch.nn as nn

from Conv1d import Conv1d
from Conv1dBank import Conv1dBank
from BatchNorm1d import BatchNorm1d
from MaxPool1d import MaxPool1d
from Highway import Highway

from Hyperparameters import Hyperparameters as hp


class DecoderCBHG(nn.Module):
    '''
    input:
        inputs: [N, T/r, n_mels * r]
    output:
        outputs: [N, T, E]
        hidden: [2, N, E//2]
    '''

    def __init__(self):
        super().__init__()

        self.conv1d_bank = nn.Sequential(
            Conv1dBank(K=hp.decoder_K, in_channels=hp.n_mels, out_channels=hp.E // 2),
            MaxPool1d(kernel_size=2)
        )

        self.projections = nn.Sequential(
            Conv1d(in_channels=hp.decoder_K * hp.E // 2, out_channels=hp.E, kernel_size=3),
            BatchNorm1d(hp.E),
            nn.ReLU(),
            Conv1d(in_channels=hp.E, out_channels=hp.n_mels, kernel_size=3),
            BatchNorm1d(hp.n_mels)
        )

        self.highways = nn.ModuleList()
        for i in range(hp.num_highways):
            self.highways.append(Highway(in_features=hp.n_mels, out_features=hp.n_mels))

        self.gru = nn.GRU(input_size=hp.n_mels, hidden_size=hp.E // 2, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, inputs, prev_hidden=None):
        inputs = inputs.view(inputs.size(0), -1, hp.n_mels)  # [N, T, n_mels]

        # conv1d bank
        outputs = self.conv1d_bank(inputs)  # [N, T, E//2 * K]

        outputs = self.projections(outputs)

        outputs = outputs + inputs  # residual connect  [N, T, n_mels]

        # highway net
        for layer in self.highways:
            outputs = layer(outputs)  # [N, T, n_mels]

        # bidirection gru
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)  # outputs: [N, T, E]

        return outputs, hidden