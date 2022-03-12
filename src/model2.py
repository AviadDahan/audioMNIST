import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio
import numpy as np
import librosa
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

class audioMNIST (Dataset):
    
    def __init__(self, audio_dir, target_sample_rate=16000, num_samples = 16250, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), add_noise = False, personal = True):
        self.personal = personal
        self.audio_dir = audio_dir
        self.device = device
        # self._create_ann()
        self.transformation = torchaudio.transforms.MelSpectrogram(target_sample_rate,
                                                      n_fft=512,    
                                                      hop_length=256, 
                                                      n_mels=64).to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.add_noise = add_noise
        # self.vad = torchaudio.transforms.Vad(self.target_sample_rate, trigger_level=4, boot_time = 0.3, allowed_gap = 0.1,
        #                                      trigger_time = 1, search_time = 1, pre_trigger_time=1, noise_reduction_amount=2,
        #                                      noise_up_time=0.8, noise_down_time=1).to(self.device)
        #                                   #  hp_filter_freq=10,
                                            #  lp_filter_freq = 8000,
                                            #  hp_lifter_freq=5,
                                            #  lp_lifter_freq=10000).to(self.device)# , allowed_gap=0.01,
                                            # #  trigger_time=1, boot_time=0.01, search_time=0.01).to(self.device)64000
        
    
    def __len__(self):
        return len(self.annotations[0])
        
    def __getitem__(self,index):
        # sample_path = self._get_sample_path(index)
        # label = self._get_sample_label(index)
#         label = label.to(self.device)
        signal, sr = torchaudio.load(self.audio_dir)
        signal = signal.to(self.device)
        # sr = sr.to(self.device)
        signal = self._mix_down(signal)
        signal = self._resample(signal, sr)
        if self.add_noise:
          signal = self._add_noise(signal)
        signal = signal/signal.norm(p=2)
        # signal = self.vad(signal)
        # signal = torchaudio.functional.lowpass_biquad(signal, self.target_sample_rate,1500).to(device)
        signal = torch.Tensor((librosa.effects.trim(signal[0].cpu().numpy(), top_db = 30))[0]).unsqueeze(dim=0).to(device)
        # print(signal)
        
        signal = self._normalize_length(signal)
        trans_signal = self.transformation(signal)
        return signal, trans_signal

   
    def _normalize_length(self, signal):
        if signal.shape[1]>self.num_samples:
            signal = signal[:,:self.num_samples]
        elif signal.shape[1]<self.num_samples:
            pad_num = self.num_samples-signal.shape[1]#last dim padding
            signal = F.pad(signal, (0,pad_num))
        return signal
            
            
    def _resample(self, signal, sr):
        if (sr != self.target_sample_rate):
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(device)
            signal = resampler(signal)
        return signal

    def _add_noise(self,signal):
      if self.personal:
        noise_path = os.path.abspath(os.path.join('/content/drive/MyDrive/PyProjects/audio/MNIST2/MNIST2/data', "..", "noise", "whitenoisegaussian.wav"))
      else:
        noise_path = os.path.abspath(os.path.join('/content/drive/MyDrive/PyProjects/audio/MNIST2', "noise", "whitenoisegaussian.wav"))
      noise, noise_sr = torchaudio.load(noise_path)
      noise = noise.to(self.device)
      noise = self._resample(noise, noise_sr)
      
      noise = noise[:, :signal.shape[1]]
      signal_power = signal.norm(p=2)
      noise_power = noise.norm(p=2)

      snr_db = 30                  # [20, 10, 3]:
      snr = math.exp(snr_db / 10)
      scale = snr * noise_power / signal_power
      noisy_signal = (scale * signal + noise) / 2
      # noisy_signal = noisy_signal/noisy_signal.norm(p=2)
      return noisy_signal

    
    def _mix_down(self, signal):
        if (signal.shape[0] > 1):
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _create_ann(self):
        Ann = [[],[],[]]
        for subdir, dirs, files in os.walk(self.audio_dir):
#             print(subdir)
            for file in files:
                if file.endswith('wav'):
                    Ann[0].append(file) ##name
                    Ann[1].append(file.split('_')[1]) #folder
                    Ann[2].append(file.split('_')[0]) #class
        Ann = np.asarray(Ann)
        self.annotations = Ann

    def shuffle(self):
      perm = torch.randperm(len(self.annotations[0]))
      self.annotations= self.annotations[:,perm]
      pass    
    
    def _get_sample_path(self, index):
        folder = self.annotations[1][index]
        filename = self.annotations[0][index]
        path = os.path.join(self.audio_dir,folder,filename)
        return path
    
    def _get_sample_label(self, index):
        label = (int(self.annotations[2][index]))
        return label


        

class Lenet5(nn.Module):

    def __init__(self, drop = False, BN = False):
        super().__init__()

        self.drop = drop
        self.BN = BN
        self.c0 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1,padding = (1,1))
        self.c0act = nn.ReLU()
        self.c0pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.c1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, stride=1)#,padding = (2,2)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.batchnorm2 = nn.BatchNorm2d(num_features = 6)
        self.s2act= nn.ReLU()
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.s4 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.batchnorm4 = nn.BatchNorm2d(num_features = 16)
        self.s4act = nn.ReLU()
        # self.s5 = nn.Linear(in_features=400, out_features=120)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride =1)
        self.batchnorm5 = nn.BatchNorm1d(num_features = 120)
        self.c5act = nn.ReLU()
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.batchnorm6 = nn.BatchNorm1d(num_features = 84)
        self.f6act = nn.ReLU()
        self.output = nn.Linear(in_features=84, out_features=10)
        
        self.dropout = nn.Dropout(p=0.1)

    def load_model(self, state_dict_location):
        state_dict = torch.load(state_dict_location,map_location=torch.device(device))
        self.load_state_dict(state_dict)

    def forward(self, x):

        if self.drop:
          x = self.c0(x)
          x = self.c0act(x)
          x = self.c0pool(x)
          x = self.c1(x)
          # print(x.size())
          x = self.s2(x)
          x = self.dropout(self.s2act(x))
          x = self.c3(x)
          x = self.s4(x)
          x = self.dropout(self.s4act(x))
          x = self.c5(x)
          x = x.view(x.shape[0], -1)
          x = self.dropout(self.c5act(x))
          x = self.f6(x)
          x = self.dropout(self.f6act(x))
          x = self.output(x)
       
        elif self.BN:
          x = self.c0(x)
          x = self.c0act(x)
          x = self.c0pool(x)
          x = self.c1(x)
          # print(x.size())
          x = self.s2(x)
          # print(x.size())
          x = self.batchnorm2(x)
          x = self.s2act(x)
          x = self.c3(x)
          x = self.s4(x)
          x = self.batchnorm4(x)
          x = self.s4act(x)
          x = self.c5(x)
          x = x.view(x.shape[0], -1)
          x = self.batchnorm5(x)
          x = self.c5act(x)
          x = self.f6(x)
          x = self.batchnorm6(x)
          x = self.f6act(x)
          x = self.output(x)
       
        else:
          x = self.c0(x)
          x = self.c0act(x)
          x = self.c0pool(x)
          # print(x.size())
          x = self.c1(x)
          x = self.s2(x)
          x = self.s2act(x)
          x = self.c3(x)
          x = self.s4(x)
          x = self.s4act(x)
          x = self.c5(x)
          # print(x.size())
          x = x.view(x.shape[0], -1)
          x = self.c5act(x)
          x = self.f6(x)
          x = self.f6act(x)
          x = self.output(x)


        
        return x