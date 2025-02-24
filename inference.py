# -*- encoding: utf-8 -*-
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
import cv2
import numpy as np
from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dehazeformer-b', type=str, help='model name')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
    parser.add_argument('--data_dir', default='D:/workspace/dehaze_dataset', type=str, help='path to dataset')
    parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
    parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
    parser.add_argument('--dataset', default='RESIDE-6K', type=str, help='dataset name')
    parser.add_argument('--exp', default='reside6k', type=str, help='experiment setting')
    parser.add_argument('--method', default='optim', type=str, choices=['orig', 'optim'], help='use algorithm')
    args = parser.parse_args()
    return args


def test(test_loader, network, result_dir, device):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    torch.cuda.empty_cache()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')
    for idx, batch in enumerate(test_loader):
        input = batch['source'].to(device)
        target = batch['target'].to(device)
        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input)
            if isinstance(output, list):
                output = output[-1]
            output = output.clamp_(-1, 1)
            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))     # Zhou Wang
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))
        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.close()
    os.rename(os.path.join(result_dir, 'results.csv'), os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


class DeHaze():
    """define DeHaze class"""
    def __init__(self, args):
        super(DeHaze, self).__init__()
        self.args = args
        if args.method == 'orig':
            from models.dehazeformer import dehazeformer_b
            self.weight_file = "./ckpt/weights/reside6k/dehazeformer-b_epoch90_psnr29.7893.pth"
            self.network = dehazeformer_b()
        elif args.method == 'optim':
            from models.dehaze_optim_v4 import Restormer
            self.weight_file = "./ckpt_optim_v4/weights/reside6k/dehazeformer-b_epoch246_psnr30.2593.pth"
            self.network = Restormer()
        else:
            raise ValueError("arguments method is wrong.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.weight_file, map_location=self.device)
        self.network.load_state_dict(checkpoint['state_dict'])
        self.network.to(self.device)
        self.network.eval()

        dataset_dir = os.path.join(args.data_dir, args.dataset)
        test_dataset = PairLoader(dataset_dir, sub_dir='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    def predict(self, image_file=None, only_single=True):
        '''define function'''
        result_dir = os.path.join(self.args.result_dir, self.args.dataset, self.args.method)
        if only_single and (image_file is not None):
            filename = os.path.basename(image_file)
            img = cv2.imread(image_file)
            source_h, source_w, c = img.shape
            img = img[:, :, ::-1].astype('float32') / 255.0
            img = img * 2 - 1
            source_h_new = int(source_h / 16) * 16
            source_w_new = int(source_w / 16) * 16
            source_img_new = cv2.resize(img, dsize=(source_w_new, source_h_new), interpolation=cv2.INTER_AREA)
            source_img_new = np.transpose(source_img_new, axes=[2, 0, 1]).copy()
            input = torch.from_numpy(source_img_new)
            input = input.unsqueeze(0)
            with torch.no_grad():
                output = self.network(input.to(self.device))
                if isinstance(output, list):
                    output = output[-1]
                output = output.clamp_(-1, 1)
                output = output * 0.5 + 0.5
            out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
            out_img = np.round((out_img[:, :, ::-1].copy() * 255.0)).astype('uint8')
            out_resize_img = cv2.resize(out_img, dsize=(source_w, source_h), interpolation=cv2.INTER_AREA)
            save_dir = os.path.join(result_dir, "single_images")
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, filename), out_resize_img)
        else:
            test(test_loader=self.test_loader, network=self.network, result_dir=result_dir, device=self.device)



if __name__ == '__main__':

    ## 设置参数
    args = set_args()
    ## 初始化去雾类
    DEHAZE = DeHaze(args=args)
    ## 单图测试
    image_file = "D:/workspace/dehaze_dataset/RESIDE-6K/test/hazy/0AAA.jpg"
    DEHAZE.predict(image_file=image_file, only_single=True)