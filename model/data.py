import os
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import numpy as np
import warnings
from torchaudio.transforms import MelSpectrogram
from util import load_index, get_frames, qtile_nomalize

clip_len = 1.0
SAMPLE_RATE = 8000

class NeuralfpDataset(Dataset):
    def __init__(self, path, n_frames, offset=0.2, norm=0.95, transform=None, train=False):
        self.path = path
        self.transform = transform
        self.train = train
        self.norm = norm
        self.offset = offset 
        self.n_frames = n_frames
        self.filenames = load_index(path)
        self.ignore_idx = []
  
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        datapath = os.path.join(self.path, self.filenames[str(idx)])
        try:
            audio, sr = torchaudio.load(datapath)
        except Exception:
            print("Error loading:" + self.filenames[str(idx)])
            self.ignore_idx.append(idx)
            # self.filenames.pop(str(idx))
            return self[idx+1]

        audio_mono = audio.mean(dim=0)
        if self.norm is not None:
            audio_mono = qtile_nomalize(audio_mono, q=self.norm)
        # print(f"audio length ----> {len(audioData)}")
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio_resampled = resampler(audio_mono)    # Downsampling
        spec = MelSpectrogram(sample_rate=SAMPLE_RATE, win_length=1024, hop_length=256, n_fft=2048, n_mels=256)    

        clip_frames = int(SAMPLE_RATE*clip_len)
        
        if len(audio_resampled) <= clip_frames:
            self.ignore_idx.append(idx)
            # self.filenames.pop(str(idx))
            return self[idx + 1]
        
        #   For training pipeline, output augmented spectrograms of a random frame of the audio
        if self.train:
            offset_mod = int(SAMPLE_RATE*(clip_frames+self.offset))
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
    
            X_j = spec(x_j)
            X_j = torchaudio.transforms.AmplitudeToDB()(X_j)
            X_j = F.pad(X_j, (self.n_frames - X_j.size(-1), 0))

            return torch.unsqueeze(X_i, 0), torch.unsqueeze(X_j, 0)
        
        #   For validation, output list of spectrograms of consecutive (overlapping) frames
        else:
            frame_length = int(SAMPLE_RATE*1.0)
            hop_length = int(SAMPLE_RATE*1.0/2)
            framed_audio = get_frames(audio_resampled, frame_length, hop_length)
            list_of_specs = []
            for frame in framed_audio:
                  X = spec(frame)
                  X = torchaudio.transforms.AmplitudeToDB()(X)
                  X = F.pad(X, (self.n_frames - X.size(-1), 0))
                  X = torch.unsqueeze(X, 0)
                  list_of_specs.append(X)
            return torch.unsqueeze(torch.cat(list_of_specs),1), self.filenames[str(idx)]
    
    def __len__(self):
        return len(self.filenames)