from audiomentations import Compose,Shift,PitchShift,TimeStretch,AddImpulseResponse,FrequencyMask,TimeMask,ClippingDistortion,AddBackgroundNoise,Gain
import numpy as np
import os
import random
import librosa

class TransformNeuralfp:
    
    def __init__(self, ir_dir, noise_dir, sample_rate):
        self.sample_rate = sample_rate
        self.ir_dir = ir_dir
        self.train_transform_i = Compose([
            # Shift(min_fraction=-0.1, max_fraction=0.1, rollover=False),
            # PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # AddImpulseResponse(ir_path=ir_dir, p=0.6),
            FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.5,p=0.8),
            TimeMask(min_band_part=0.1, max_band_part=0.5),
            # ClippingDistortion(),
            # AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=0, max_snr_in_db=10,p=0.9),
            # Gain(),
            # Mp3Compression()
            ])
        
        self.train_transform_j = Compose([
            # Shift(min_fraction=-0.1, max_fraction=0.1, rollover=False),
            # PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
            AddImpulseResponse(ir_path=ir_dir, p=0.8),
            FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.5,p=0.8),
            TimeMask(min_band_part=0.1, max_band_part=0.5),
            # ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=10),
            AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=0, max_snr_in_db=7,p=0.8),
            # Gain(),
            # Mp3Compression()
            ])
    def irconv(self, x, p):
        ir_dir = self.ir_dir
        if random.random() < p:
            r1 = random.randrange(len(os.listdir(ir_dir)))
            fpath = os.path.join(ir_dir, os.listdir(ir_dir)[r1])
            x_ir, fs = librosa.load(fpath, sr=None)
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
        x_j = self.irconv(x_j, p=0.8)
        return self.train_transform_i(x_i, sample_rate=self.sample_rate), self.train_transform_j(x_j, sample_rate=self.sample_rate)