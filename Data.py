import os
import unicodedata
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import *

class TacotronDataset(Dataset):
    '''
    text: [T_x]
    mel: [T_y/r, n_mels*r]
    '''
    def __init__(self, r=slice(0, None)):
        fpaths, texts = get_LJ_data(hp.data, r)
        self.fpaths = fpaths
        self.texts = texts
    
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        _, mel, _ = load_spectrograms(self.fpaths[idx])
        mel = torch.from_numpy(mel)
        GO_mel = torch.zeros(1, mel.size(1))  # GO frame
        mel = torch.cat([GO_mel, mel], dim=0)
        text = self.texts[idx]
        return text, mel
    
    @staticmethod
    def collate_fn(batch):
        batch_text = [x for x, _ in batch]
        batch_mel = [y for _, y in batch]

        batch_text_pad = pad_sequence(batch_text, batch_first=True)
        lens_text = [len(text) for text in batch_text]

        batch_mel_pad = pad_sequence(batch_mel, batch_first=True)
        lens_mel = [len(mel) for mel in batch_mel]

        return batch_text_pad, batch_mel_pad, torch.tensor(lens_text).type(torch.int), torch.tensor(lens_mel).type(torch.int)
        


def get_LJ_data(data_dir, r):
    path = os.path.join(data_dir, 'metadata.csv')
    data_dir = os.path.join(data_dir, 'wavs')
    wav_paths = []
    texts = []
    with open(path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split('|')
            wav_paths.append(os.path.join(data_dir, items[0] + '.wav'))
            text = items[1]
            text = text_normalize(text) + 'E'
            text = [hp.char2idx[c] for c in text]
            text = torch.Tensor(text).type(torch.LongTensor)
            texts.append(text)

    return wav_paths[r], texts[r]

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


if __name__ == '__main__':
    dataset = TacotronDataset()
    loader = DataLoader(dataset=dataset, batch_size=8, collate_fn=TacotronDataset.collate_fn)

    for batch in loader:
        text, mel, text_len, mel_len = batch
        print(text[0])
        print(mel.size())
        print(text_len, mel_len)
        break