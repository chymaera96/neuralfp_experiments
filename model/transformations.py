# from torch_audiomentations import Compose,AddBackgroundNoise, ApplyImpulseResponse
from audiomentations import Compose,AddBackgroundNoise, ApplyImpulseResponse
import numpy as np
import os
import random
import librosa
import warnings

class TransformNeuralfp:
    
    def __init__(self, ir_dir, noise_dir, sample_rate):
        self.sample_rate = sample_rate
        self.ir_dir = ir_dir
        # self.train_transform_i = Compose([
        #     FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.5,p=0.8),
        #     TimeMask(min_band_part=0.1, max_band_part=0.5),

        #     ])
        
        self.train_transform_j = Compose([
            ApplyImpulseResponse(ir_path=self.ir_dir, p=0.5, leave_length_unchanged=True),
            AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=0, max_snr_in_db=20,p=0.8),

            ])
        
    def irconv(self, x, p):
        ir_dir = self.ir_dir
        if random.random() < p:
            r1 = random.randrange(len(os.listdir(ir_dir)))
            fpath = os.path.join(ir_dir, os.listdir(ir_dir)[r1])
            x_ir, fs = librosa.load(fpath, sr=self.sample_rate)
            # x_ir = x_ir.mean(axis=0)
            fftLength = np.maximum(len(x), len(x_ir))
            X = np.fft.fft(x, n=fftLength)
            X_ir = np.fft.fft(x_ir, n=fftLength)
            x_aug = np.fft.ifft(np.multiply(X_ir, X))[0:len(x)].real
            if np.max(np.abs(x_aug)) == 0:
                pass
            else:
                x_aug = x_aug / np.max(np.abs(x_aug))  # Max-normalize
        
        else: 
            x_aug = x
        
        return x_aug.astype(np.float32)
            
    def __call__(self, x_i, x_j):
        # x_j = self.irconv(x_j, p=0.8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_j = self.train_transform_j(x_j, sample_rate=self.sample_rate)
        return x_i, x_j