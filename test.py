import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import faiss
import json
import shutil

from util import load_ckp, save_ckp, create_train_set
from model.transformations import TransformNeuralfp
from model.data import NeuralfpDataset
from eval import get_index, load_memmap_data, eval_faiss
from torch.utils.data.sampler import SubsetRandomSampler



# Directories
root = os.path.dirname(__file__)

ir_dir = os.path.join(root,'data/augmentation_datasets/ir_filters')
noise_dir = os.path.join(root,'data/augmentation_datasets/noise')

parser = argparse.ArgumentParser(description='Neuralfp Testing')
parser.add_argument('--test_dir', default='', type=str)
parser.add_argument('--fp_dir', default='fingerprints', type=str)
parser.add_argument('--ckp', default='', type=str)
parser.add_argument('--query_lens', default='1 3 5 9 11 19', type=str)
parser.add_argument('--n_dummy_db', default=500, type=int)
parser.add_argument('--n_query_db', default=20, type=int)
parser.add_argument('--compute_fp', default=True, type=bool)
parser.add_argument('--small_test', default=False, type=bool)
parser.add_argument('--nb', default=False, type=bool)


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def create_fp_db(dataloader, model, output_root_dir, save=True):
    fp_q = []
    fp_db = []
    print("=> Creating query and db fingerprints...")
    for idx, (db,q) in enumerate(dataloader):
        splits = zip(db[0], q[0])
        for x_i, x_j in splits:
            x_i = torch.unsqueeze(x_i,0).to(device)
            x_j = torch.unsqueeze(x_j,0).to(device)

            with torch.no_grad():
                _, _, z_i, z_j= model(x_i,x_j)

            z_i = z_i.detach().cpu().numpy()
            z_j = z_j.detach().cpu().numpy()

            fp_db.append(z_i)
            fp_q.append(z_j)

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z_i.shape}")
    
    arr_shape = (len(fp_db), z_i.shape[-1])
    print(arr_shape)
    fp_db = np.concatenate(fp_db)
    fp_q = np.concatenate(fp_q)

    arr_q = np.memmap(f'{output_root_dir}/query.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    print(len(fp_q))

    arr_q[:] = fp_q[:]
    arr_q.flush(); del(arr_q)   #Close memmap

    np.save(f'{output_root_dir}/query_shape.npy', arr_shape)

    arr_db = np.memmap(f'{output_root_dir}/db.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr_db[:] = fp_db[:]
    arr_db.flush(); del(arr_db)   #Close memmap

    np.save(f'{output_root_dir}/db_shape.npy', arr_shape)

def create_dummy_db(dataloader, model, output_root_dir, fname='dummy_db', save=True):
    fp = []
    print("=> Creating dummy fingerprints...")
    for idx, (db,q) in enumerate(dataloader):
        splits = zip(db[0], q[0])
        for x_i, x_j in splits:
            x_i = torch.unsqueeze(x_i,0).to(device)

            with torch.no_grad():
                _, _, z_i, _= model(x_i,x_i)

            z_i = z_i.detach().cpu().numpy()

            fp.append(z_i)

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z_i.shape}")
    
    arr_shape = (len(fp), z_i.shape[-1])
    fp = np.concatenate(fp)

    arr = np.memmap(f'{output_root_dir}/{fname}.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr[:] = fp[:]
    arr.flush(); del(arr)   #Close memmap

    np.save(f'{output_root_dir}/{fname}_shape.npy', arr_shape)


def main():

    args = parser.parse_args()
    # json_dir = load_index(data_dir)

    # Hyperparameters
    random_seed = 42
    shuffle_dataset =True
            
    print("Loading Model...")
    model = None # Model

       
    if os.path.isfile(args.ckp):
        print("=> loading checkpoint '{}'".format(args.ckp))
        checkpoint = torch.load(args.ckp)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.ckp))



    print("Creating dataloaders ...")
    dataset = NeuralfpDataset(path=args.test_dir, transform=TransformNeuralfp(ir_dir=ir_dir, noise_dir=noise_dir,sample_rate=22050), train=False)


    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split1 = args.n_dummy_db
    split2 = args.n_query_db
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    dummy_indices, query_db_indices = indices[:split1], indices[split1: split1 + split2]
    print(f"Length of indices {len(dummy_indices)} {len(query_db_indices)}")

    dummy_db_sampler = SubsetRandomSampler(dummy_indices)
    query_db_sampler = SubsetRandomSampler(query_db_indices)

    

    dummy_db_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=dummy_db_sampler)
    
    query_db_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=query_db_sampler)


    if not os.path.exists(args.fp_dir):
        os.mkdir(args.fp_dir)

    if args.compute_fp == True:
        create_fp_db(query_db_loader, model, args.fp_dir)
        create_dummy_db(dummy_db_loader, model, args.fp_dir)

    if args.small_test:
        index_type = 'l2'
    else:
        index_type = 'ivfpq'
    eval_faiss(emb_dir=args.fp_dir, test_ids='all', test_seq_len=args.query_lens, index_type=index_type)




if __name__ == '__main__':
    main()