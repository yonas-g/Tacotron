import torch
import torch.nn as nn
import torch.nn.functional as F

from Hyperparameters import Hyperparameters as hp

class Attention(nn.Module):
    '''
    input:
        inputs: [N, T_y, E//2] mels directly from audio (80-band processed)
        memory: [N, T_x, E] encoded text from the encoder

    output:
        attn_weights: [N, T_y, T_x]
        outputs: [N, T_y, E]
        hidden: [1, N, E]

    T_x --- character len
    T_y --- spectrogram len
    '''

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=hp.E // 2, hidden_size=hp.E, batch_first=True, bidirectional=False)
        self.W = nn.Linear(in_features=hp.E, out_features=hp.E, bias=False)
        self.U = nn.Linear(in_features=hp.E, out_features=hp.E, bias=False)
        self.v = nn.Linear(in_features=hp.E, out_features=1, bias=False)

    def forward(self, inputs, memory, prev_hidden=None):
        T_x = memory.size(1)
        T_y = inputs.size(1)

        # inputs = torch.cat([inputs[:, 0, :].unsqueeze(1), inputs[:, :-1, :]], 1)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(inputs, prev_hidden)  # outputs: [N, T_y, E]  hidden: [1, N, E]
        w = self.W(outputs).unsqueeze(2).expand(-1, -1, T_x, -1)  # [N, T_y, T_x, E]
        u = self.U(memory).unsqueeze(1).expand(-1, T_y, -1, -1)  # [N, T_y, T_x, E]
        attn_weights = self.v(torch.tanh(w + u).view(-1, hp.E)).view(-1, T_y, T_x)
        attn_weights = F.softmax(attn_weights, 2)

        return attn_weights, outputs, hidden