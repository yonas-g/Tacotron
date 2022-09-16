import torch
import torch.nn as nn

from PreNet import PreNet
from Attention import Attention
from DecoderCBHG import DecoderCBHG

from Hyperparameters import Hyperparameters as hp


class Decoder(nn.Module):
    '''
    input:
        inputs --- [N, T_y/r, n_mels * r]
        memory --- [N, T_x, E]
    output:
        mels   --- [N, T_y/r, n_mels*r]
        mags --- [N, T_y, 1+n_fft//2]
        attn_weights --- [N, T_y/r, T_x]
    '''

    def __init__(self):
        super().__init__()
        self.prenet = PreNet(hp.n_mels)
        self.attn_rnn = Attention()
        self.attn_projection = nn.Linear(in_features=2 * hp.E, out_features=hp.E)
        self.gru1 = nn.GRU(input_size=hp.E, hidden_size=hp.E, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(input_size=hp.E, hidden_size=hp.E, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(in_features=hp.E, out_features=hp.n_mels * hp.r)
        self.cbhg = DecoderCBHG()  # Deng
        self.fc2 = nn.Linear(in_features=hp.E, out_features=1 + hp.n_fft // 2)  # Deng

    def forward(self, inputs, memory):
        if self.training:
            # prenet
            outputs = self.prenet(inputs)  # [N, T_y/r, E//2]

            attn_weights, outputs, attn_hidden = self.attn_rnn(outputs, memory)

            attn_apply = torch.bmm(attn_weights, memory)  # [N, T_y/r, E]
            attn_project = self.attn_projection(torch.cat([attn_apply, outputs], dim=2))  # [N, T_y/r, E]

            # GRU1
            self.gru1.flatten_parameters()
            outputs1, gru1_hidden = self.gru1(attn_project)  # outputs1--[N, T_y/r, E]  gru1_hidden--[1, N, E]
            gru_outputs1 = outputs1 + attn_project  # [N, T_y/r, E]
            # GRU2
            self.gru2.flatten_parameters()
            outputs2, gru2_hidden = self.gru2(gru_outputs1)  # outputs2--[N, T_y/r, E]  gru2_hidden--[1, N, E]
            gru_outputs2 = outputs2 + gru_outputs1

            # generate log melspectrogram
            mels = self.fc1(gru_outputs2)  # [N, T_y/r, n_mels*r]

            # CBHG
            out, cbhg_hidden = self.cbhg(mels)  # out -- [N, T_y, E]

            # generate linear spectrogram
            mags = self.fc2(out)  # out -- [N, T_y, 1+n_fft//2]

            return mels, mags, attn_weights

        else:
            # inputs = Go_frame  [1, 1, n_mels*r]
            attn_hidden = None
            gru1_hidden = None
            gru2_hidden = None

            mels = []
            mags = []
            attn_weights = []
            for i in range(hp.max_Ty):
                inputs = self.prenet(inputs)
                attn_weight, outputs, attn_hidden = self.attn_rnn(inputs, memory, attn_hidden)
                attn_weights.append(attn_weight)  # attn_weight: [1, 1, T_x]
                attn_apply = torch.bmm(attn_weight, memory)  # [1, 1, E]
                attn_project = self.attn_projection(torch.cat([attn_apply, outputs], dim=-1))  # [1, 1, E]

                # GRU1
                self.gru1.flatten_parameters()
                outputs1, gru1_hidden = self.gru1(attn_project, gru1_hidden)  # outputs1--[1, 1, E]  gru1_hidden--[1, 1, E]
                outputs1 = outputs1 + attn_project  # [1, T_y/r, E]
                # GRU2
                self.gru2.flatten_parameters()
                outputs2, gru2_hidden = self.gru2(outputs1, gru2_hidden)  # outputs2--[1, T_y/r, E]  gru2_hidden--[1, 1, E]
                outputs2 = outputs2 + outputs1

                # generate log melspectrogram
                mel = self.fc1(outputs2)  # [1, 1, n_mels*r]
                inputs = mel[:, :, -hp.n_mels:]  # get last frame
                mels.append(mel)

            mels = torch.cat(mels, dim=1)  # [1, max_iter, n_mels*r]
            attn_weights = torch.cat(attn_weights, dim=1)  # [1, T, T_x]

            out, cbhg_hidden = self.cbhg(mels)
            mags = self.fc2(out)

            return mels, mags, attn_weights

