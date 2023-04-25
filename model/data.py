import os
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
from librosa.feature import melspectrogram
import warnings
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking
import torch.nn as nn
import warnings


from util import load_index, get_frames, qtile_normalize

clip_len = 2.0
SAMPLE_RATE = 22050

class NeuralfpDataset(Dataset):
    def __init__(self, path, n_frames=240, offset=0.2, norm=0.95, transform=None, train=False):
        self.path = path
        self.transform = transform
        self.train = train
        self.norm = norm
        self.offset = offset 
        self.n_frames = n_frames
        self.filenames = load_index(path)
        self.spec_aug = nn.Sequential(
            TimeMasking(time_mask_param=80),
            FrequencyMasking(freq_mask_param=64)
        )

        self.ignore_idx = []
  
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        datapath = self.filenames[str(idx)]
        try:
            audio, sr = torchaudio.load(datapath)

        except Exception:

            print("Error loading:" + self.filenames[str(idx)])
            self.ignore_idx.append(idx)
            # self.filenames.pop(str(idx))
            return self[idx+1]

        audio_mono = audio.mean(dim=0)
        if self.norm is not None:
            audio_mono = qtile_normalize(audio_mono, q=self.norm)
        # print(f"audio length ----> {len(audioData)}")
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio_resampled = resampler(audio_mono)    # Downsampling
        spec = MelSpectrogram(sample_rate=22050, win_length=740, hop_length=185, n_fft=740, n_mels=128)      

        clip_frames = int(SAMPLE_RATE*clip_len)
        
        if len(audio_resampled) <= clip_frames:
            self.ignore_idx.append(idx)
            # self.filenames.pop(str(idx))
            return self[idx + 1]
        
        #   For training pipeline, output augmented spectrograms of a random frame of the audio
        if self.train:
            offset_mod = int(SAMPLE_RATE*(self.offset) + clip_frames)
            if len(audio_resampled) < offset_mod:
                print(len(audio_resampled), offset_mod)
            r = np.random.randint(0,len(audio_resampled)-offset_mod)
            ri = np.random.randint(0,offset_mod - clip_frames)
            rj = np.random.randint(0,offset_mod - clip_frames)
            clip = audio_resampled[r:r+offset_mod]
            org = clip[ri:ri+clip_frames]
            rep = clip[rj:rj+clip_frames]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_i, x_j = self.transform(org.numpy(), rep.numpy())
            x_i = torch.from_numpy(x_i)
            x_j = torch.from_numpy(x_j)
            
            X_i = spec(x_i)
            X_i = torchaudio.transforms.AmplitudeToDB()(X_i)
            X_i = F.pad(X_i, (self.n_frames - X_i.size(-1), 0))
            X_i = self.spec_aug(X_i)
    
            X_j = spec(x_j)
            X_j = torchaudio.transforms.AmplitudeToDB()(X_j)
            X_j = F.pad(X_j, (self.n_frames - X_j.size(-1), 0))
            X_j = self.spec_aug(X_j)
            # print(X_j.shape)


            return torch.unsqueeze(X_i.T, 0), torch.unsqueeze(X_j.T, 0)
        
        else:
            frame_length = int(SAMPLE_RATE*clip_len)
            hop_length = int(SAMPLE_RATE*clip_len/2)
            framed_audio = get_frames(audio_resampled, frame_length, hop_length)
            assert len(framed_audio) > 0 
            list_of_specs_i = []
            list_of_specs_j = []

            for frame in framed_audio:
                if self.transform:
                    f_i, f_j = self.transform(frame.numpy(), frame.numpy())
                    f_i = torch.from_numpy(f_i)
                    f_j = torch.from_numpy(f_j)

                else:
                    f_i = frame
                    f_j = frame.clone()

                X_i = spec(f_i)
                X_i = torchaudio.transforms.AmplitudeToDB()(X_i)
                if X_i.size(-1) < self.n_frames:
                    X_i = F.pad(X_i, (self.n_frames - X_i.size(-1), 0))
                X_i = torch.unsqueeze(X_i.T, 0)
                list_of_specs_i.append(X_i)

                X_j = spec(f_j)
                X_j = torchaudio.transforms.AmplitudeToDB()(X_j)
                if X_j.size(-1) < self.n_frames:
                    X_j = F.pad(X_j, (self.n_frames - X_j.size(-1), 0))
                X_j = torch.unsqueeze(X_j.T, 0)
                list_of_specs_j.append(X_j)

                assert len(list_of_specs_i) > 0 

            return torch.unsqueeze(torch.cat(list_of_specs_i),1), torch.unsqueeze(torch.cat(list_of_specs_j),1)
    
    def __len__(self):
        return len(self.filenames)