import os
import torch
import librosa
import numpy as np
import json
import glob

def load_index(data_dir, ext=['wav','mp3']):
    dataset = {}

    print(f"=>Loading indices from {data_dir}")
    json_path = os.path.join(data_dir, data_dir.split('/')[-1] + ".json")

    for idx,fpath in enumerate(glob.iglob(os.path.join(data_dir,'**/*.*'), recursive=True)):
        if fpath.split('.')[-1] in ext: 
            dataset[str(idx)] = fpath
        
    with open(json_path, 'w') as fp:
        json.dump(dataset, fp)

    return dataset

def get_frames(y, frame_length, hop_length):
    # frames = librosa.util.frame(y.numpy(), frame_length, hop_length, axis=0)
    frames = y.unfold(0, size=frame_length, step=hop_length)
    return frames

def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + torch.quantile(y,q=q))


def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['loss']

def save_ckp(state,epoch,model_name,model_folder):
    if not os.path.exists(model_folder): 
        os.mkdir(model_folder)
    torch.save(state, "{}/model_{}_epoch_{}.pth".format(model_folder,model_name,epoch))