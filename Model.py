import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder

from Hyperparameters import Hyperparameters as hp


class Tacotron(nn.Module):
    '''
    input:
        texts: [N, T_x]
        mels: [N, T_y/r, n_mels*r]
    output:
        mels --- [N, T_y/r, n_mels*r]
        mags --- [N, T_y, 1+n_fft//2]
        attn_weights --- [N, T_y/r, T_x]
    '''

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(hp.vocab), hp.E)
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, texts, mels, ref_mels):
        embedding = self.embedding(texts)  # [N, T_x, E]
        memory, encoder_hidden = self.encoder(embedding)  # [N, T_x, E]

        mels_hat, mags_hat, attn_weights = self.decoder(mels, memory)

        return mels_hat, mags_hat, attn_weights