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
        _, mel, mag = load_spectrograms(self.fpaths[idx])
        mel = torch.from_numpy(mel)
        mag = torch.from_numpy(mag)
        GO_mel = torch.zeros(1, mel.size(1))  # GO frame
        mel = torch.cat([GO_mel, mel], dim=0)
        text = self.texts[idx]
        return {'text': text, 'mel': mel, 'mag': mag}
    
    @staticmethod
    def collate_fn(batch):
        batch_text = [b["text"] for b in batch]
        batch_mel = [b["mel"] for b in batch]
        batch_mag = [b["mag"] for b in batch]

        batch_text_pad = pad_sequence(batch_text, batch_first=True)
        lens_text = [len(text) for text in batch_text]

        batch_mel_pad = pad_sequence(batch_mel, batch_first=True)
        lens_mel = [len(mel) for mel in batch_mel]

        batch_mag_pad = pad_sequence(batch_mag, batch_first=True)
        lens_mag = [len(mag) for mag in batch_mag]

        batch = {
            "text": batch_text_pad,
            "mel": batch_mel_pad,
            "mag": batch_mag_pad,
            "len_text": torch.tensor(lens_text).type(torch.int),
            "len_mel": torch.tensor(lens_mel).type(torch.int),
            "len_mag": torch.tensor(lens_mag).type(torch.int)
        }

        return batch
        


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
        print(batch["text"][0])
        print(batch['mel'].size())
        print(batch['mag'].size())
        break