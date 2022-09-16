import os
import gc
import sys
import subprocess
from time import time

import torch
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from Model import Tacotron
from Loss import TacotronLoss

from Data import *
from Hyperparameters import Hyperparameters as hp

import matplotlib.pyplot as plt
from scipy.io.wavfile import write

device = torch.device(hp.device)

def train(log_dir=0, dataset_size=None, start_epoch = 0):

    f = init_log_dir(log_dir)
    log(f, 'use {}'.format(device))
    
    model = Tacotron().to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        log(f, "Using multiple GPUs")
        log(f, f"Number of GPUs: {torch.cuda.device_count()}")
    
    if start_epoch != 0:
        model_path = os.path.join(log_dir, 'state', 'epoch{}.pt'.format(start_epoch))
        model.load_state_dict(torch.load(model_path))
        log(f, 'Load model of' + model_path)
    else:
        log(f, 'New model')

    
    # load data
    if dataset_size is None:
        train_dataset = TacotronDataset(r=slice(hp.eval_size, None))
    else:
        train_dataset = TacotronDataset(r=slice(hp.eval_size, hp.eval_size + dataset_size))

    train_loader = DataLoader(dataset=train_dataset, batch_size=hp.batch_size, collate_fn=TacotronDataset.collate_fn, num_workers=5, shuffle=True)
    
    criterion = TacotronLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader)*hp.num_epochs))
    
    if start_epoch != 0:
        opt_path = os.path.join(log_dir, 'state_opt', 'epoch{}.pt'.format(start_epoch))
        optimizer.load_state_dict(torch.load(opt_path))
        log(f, 'Load optimizer of' + opt_path)


    num_train_data = len(train_dataset)
    total_step = hp.num_epochs * num_train_data // hp.batch_size
    start_step = start_epoch * num_train_data // hp.batch_size
    step = 0
    global_step = step + start_step
    prev = beg = int(time())

    for epoch in range(start_epoch + 1, hp.num_epochs):
        
        model.train()

        for i, batch in enumerate(train_loader):

            texts = batch['text'].to(device)
            mels = batch['mel'].to(device)
            mags = batch['mag'].to(device)

            optimizer.zero_grad()

            mels_input = mels[:, :-1, :]  # shift
            mels_input = mels_input[:, :, -hp.n_mels:]  # get last frame
            ref_mels = mels[:, 1:, :]

            mels_hat, mags_hat, _ = model(texts, mels_input, ref_mels)

            mel_loss, mag_loss = criterion(mels[:, 1:, :], mels_hat, mags, mags_hat)
            loss = mel_loss + mag_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)  # clip gradients
            optimizer.step()
            scheduler.step()

            if (i + 1) % hp.log_per_batch == 0:
                now = int(time())
                use_time = now - prev
                total_time = total_step * (now - beg) // step
                left_time = total_time - (now - beg)
                left_time_h = left_time // 3600
                left_time_m = left_time // 60 % 60
                msg = 'step: {}/{}, epoch: {}, batch {}, loss: {:.3f}, mel_loss: {:.3f}, mag_loss: {:.3f}, use_time: {}s, left_time: {}h {}m'
                msg = msg.format(global_step, total_step, epoch, i + 1, loss.item(), mel_loss.item(), mag_loss.item(), use_time, left_time_h, left_time_m)

                log(f, msg)

                prev = now
        
        # save model, optimizer and evaluate
        if epoch % hp.save_per_epoch == 0 and epoch != 0:
            eval(epoch, log_dir, f, model, optimizer)
    
    log(f, 'Training Done')
    f.close()


def eval(epoch, log_dir, f, model, optimizer):
    torch.save(model.state_dict(), os.path.join(log_dir, 'state/epoch{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(log_dir, 'state_opt/epoch{}.pt'.format(epoch)))
    log(f, 'save model, optimizer in epoch{}'.format(epoch))

    model.eval()

    #for file in os.listdir(hp.ref_wav):
    wavfile = hp.ref_wav
    name, _ = os.path.splitext(hp.ref_wav.split('/')[-1])

    text, mel, ref_mels = get_eval_data(hp.eval_text, wavfile)
    text = text.to(device)
    mel = mel.to(device)
    ref_mels = ref_mels.to(device)

    mel_hat, mag_hat, attn = model(text, mel, ref_mels)

    mag_hat = mag_hat.squeeze().detach().cpu().numpy()
    attn = attn.squeeze().detach().cpu().numpy()

    fig_path = os.path.join(log_dir, 'attn/epoch{}-{}.png'.format(epoch, name))
    
    plt.imshow(attn.T, cmap='hot', interpolation='nearest')
    plt.xlabel('Decoder Steps')
    plt.ylabel('Encoder Steps')
    plt.savefig(fig_path, format='png')

    wav = spectrogram2wav(mag_hat)
    write(os.path.join(log_dir, 'wav/epoch{}-{}.wav'.format(epoch, name)), hp.sr, wav)

    log(f, 'synthesis eval wav in epoch{} model'.format(epoch))

    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    log(f, result.stdout.decode("utf-8"))


def init_log_dir(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'state')):
        os.mkdir(os.path.join(log_dir, 'state'))
    if not os.path.exists(os.path.join(log_dir, 'wav')):
        os.mkdir(os.path.join(log_dir, 'wav'))
    if not os.path.exists(os.path.join(log_dir, 'state_opt')):
        os.mkdir(os.path.join(log_dir, 'state_opt'))
    if not os.path.exists(os.path.join(log_dir, 'attn')):
        os.mkdir(os.path.join(log_dir, 'attn'))
    if not os.path.exists(os.path.join(log_dir, 'test_wav')):
        os.mkdir(os.path.join(log_dir, 'test_wav'))

    return open(os.path.join(log_dir, 'log{}.txt'.format(start_epoch)), 'w')


def empty_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":

    argv = sys.argv

    if len(argv) < 2:
        log_number = 0
        start_epoch = 0
        dataset_size = None
    else:
        log_number = int(argv[1])
        start_epoch = int(argv[3])
        
        if argv[2].lower() != 'all':
            dataset_size = int(argv[2])
        else:
            dataset_size = None
    
    train(hp.log_dir.format(log_number), dataset_size, start_epoch)
