# -*- encoding: utf-8 -*-
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
from models.dehaze_optim_v2 import Restormer

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dehazeformer-b', type=str, help='model name')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
    parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
    parser.add_argument('--checkpoint_dir', default="./ckpt/", type=str, help='path to save result in training')
    parser.add_argument('--data_dir', default='D:/workspace/dehaze_dataset', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='RESIDE-6K', type=str, help='dataset name')
    parser.add_argument('--exp', default='reside6k', type=str, help='experiment setting')
    parser.add_argument('--resume', default='', type=str, help='checkpoint model file')
    args = parser.parse_args()
    return args


def train(train_loader, network, criterion, optimizer, scaler, device):
    losses = AverageMeter()
    torch.cuda.empty_cache()
    network.train()
    train_num = len(train_loader)
    iter_num = 0
    for batch in train_loader:
        iter_num += 1
        source_img = batch['source'].to(device)
        target_img = batch['target'].to(device)
        # with autocast(args.no_autocast):
        #   output = network(source_img)
        output = network(source_img)
        loss = criterion(output[-1], target_img)
        for output_ele in output[:-1]:
            loss += criterion(output_ele, target_img)
        losses.update(loss.item())
        print("Iter [{}/{}]  Loss:{:.5f}".format(iter_num, train_num, loss.item()))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network, device):
    PSNR = AverageMeter()
    torch.cuda.empty_cache()
    network.eval()

    for batch in val_loader:
        source_img = batch['source'].to(device)
        target_img = batch['target'].to(device)
        with torch.no_grad():
            output = network(source_img)[-1].clamp_(-1, 1)
        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


def process(args):
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # network = eval(args.model.replace('-', '_'))()
    network = Restormer()
    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        network.load_state_dict(state_dict['state_dict'], strict=False)

    # network = nn.DataParallel(network).cuda()
    network.to(device)

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)

    train_dataset = PairLoader(dataset_dir, 'train', 'train', setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset, batch_size=setting['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], setting['patch_size'])
    val_loader = DataLoader(val_dataset, batch_size=setting['batch_size'], num_workers=args.num_workers, pin_memory=True)

    save_dir = os.path.join(args.checkpoint_dir, "weights", args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):
        print('==> Start training, current model name: ' + args.model)
        # print(network)
        writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, "logs", args.exp, args.model))

        best_psnr = 0
        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion, optimizer, scaler, device)
            writer.add_scalar('train_loss', loss, epoch)
            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network, device)
                print("Epoch: {}  avg_psnr: {}".format(epoch, avg_psnr))
                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + '_epoch{}_psnr{:.4f}.pth'.format(epoch, best_psnr)))

                writer.add_scalar('best_psnr', best_psnr, epoch)

    else:
        print('==> Existing trained model')
        exit(1)


if __name__ == '__main__':

    args = set_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    process(args)