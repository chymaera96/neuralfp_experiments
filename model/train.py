import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F

from util import load_ckp, save_ckp, create_train_set
from model.transformations import TransformNeuralfp
from model.data import NeuralfpDataset

from torch.utils.data.sampler import SubsetRandomSampler


# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")

device = torch.device("cuda")


parser = argparse.ArgumentParser(description='Neuralfp Training')
parser.add_argument('--data_dir', default='', type=str,
                    help='Path to training data')
parser.add_argument('--ir_dir', default='', type=str,
                    help='Path to impulse response data (augmentation)')
parser.add_argument('--noise_dir', default='', type=str,
                    help='Path to background noise data (augmentation)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--sr', default=22050, type=int,
                    help='Sampling rate ')
parser.add_argument('--ckp', default='sfnet', type=str,
                    help='checkpoint_name')


def ntxent_loss(z_i, z_j, tau=0.05):
    """
    NTXent Loss function.
    Parameters
    ----------
    z_i : torch.tensor
        embedding of original samples (batch_size x emb_size)
    z_j : torch.tensor
        embedding of augmented samples (batch_size x emb_size)
    Returns
    -------
    loss
    """
    z = torch.stack((z_i,z_j), dim=1).view(2*z_i.shape[0], z_i.shape[1])
    a = torch.matmul(z, z.T)
    a /= tau
    Ls = []
    for i in range(z.shape[0]):
        nn_self = torch.cat([a[i,:i], a[i,i+1:]])
        softmax = torch.nn.functional.log_softmax(nn_self, dim=0)
        Ls.append(softmax[i if i%2 == 0 else i-1])
    Ls = torch.stack(Ls)
    
    loss = torch.sum(Ls) / -z.shape[0]
    return loss

def train(train_loader, model, optimizer):
    loss_epoch = 0
    for idx, (x_i, x_j) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        loss = ntxent_loss(z_i, z_j)

        # if torch.count_nonzero(torch.isnan(loss)) > 0:
        #     print(z_i)
        loss.backward()

        optimizer.step()

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(train_loader)}]\t Loss: {loss.item()}")
      

        loss_epoch += loss.item()
    return loss_epoch


def main():
    args = parser.parse_args()
    data_dir = args.data_dir
    ir_dir = args.ir_dir
    noise_dir = args.noise_dir
    
    # Hyperparameters
    batch_size = 120        # PROBABLY TOO SMALL FOR SimCLR
    learning_rate = 1e-4
    num_epochs = args.epochs
    sample_rate = args.sr
    model_name = args.ckp
    random_seed = 42

    print(ir_dir)
    print(noise_dir)
    assert data_dir == os.path.join(root,"data/fma_8000")

    print("Loading dataset...")
    train_dataset = NeuralfpDataset(path=data_dir, transform=TransformNeuralfp(ir_dir=ir_dir, noise_dir=noise_dir,sample_rate=sample_rate), train=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    
    
    print("Creating new model...")
    model = None # ADD MODEL
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 500, eta_min = 1e-7)

       
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model, optimizer, scheduler, start_epoch, loss_log = load_ckp(args.resume, model, optimizer, scheduler)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    else:
        start_epoch = 0
        loss_log = []

    print("Calculating initial loss ...")
    best_loss = train(train_loader, model, optimizer)

    # training
    model.train()
    for epoch in range(start_epoch+1, num_epochs+1):
        print("#######Epoch {}#######".format(epoch))
        loss_epoch = train(train_loader, model, optimizer)
        loss_log.append(loss_epoch)
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            
            checkpoint = {
                'epoch': epoch,
                'loss': loss_log,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            save_ckp(checkpoint,epoch, model_name, model_folder)
        scheduler.step()
    
  
        
if __name__ == '__main__':
    main()