import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.distributed as dist


import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.GeleNet_models import GeleNet
from data import get_loader
from utils import clip_gradient, adjust_lr
from model.H_vmunet import H_vmunet
# from model.LightMUNet import LightMUNet
# from model.RS3Mamba import RS3Mamba
# from model.RS3att import RS3Mamba
# from model. SpikingRS314 import RS3Mamba
# from model.egeunet import EGEUNet
# from model.back2 import RS3Mamba
# from model.UNet import UNet
from model.DeeplabV3Plus import DeepLab
from model.Swin_Unet import SwinUnet
from model.ptavitssg2.ptavitssg2_dn import *
from model.RollingUnet import Rolling_Unet_L
from model.SCTransNet import SCTransNet

import pytorch_iou
import torch.nn.functional as F

torch.cuda.set_device(0)
# torch.cuda.init()
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()

# parser.add_argument('--clip', type=bool, default=False)
opt = parser.parse_args()

# build models
# model = GeleNet()
# model = UNet()
# model = H_vmunet()
# model = LightMUNet()
# model = RS3Mamba()
# model = EGEUNet()
# model = DeepLab(1)
# model = SwinUnet()
model = Rolling_Unet_L(1)







model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
image_root = r'E:\\LoveDA\\Train2\\image512\\'
gt_root = r'E:\\LoveDA\\Train2\\mask512\\'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)


CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)


def train(train_loader, model, optimizer, epoch):
    model.train()
    torch.autograd.set_detect_anomaly(True) #异常检测
    total_loss = 0.0
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        # sal, sal_sig = model(images)!!!!
        sal, sal_sig = model(images)
        # sal, sal_sig, topo = model(images)
        # 调整 sal 的尺寸 light

        # sal = sal[0] #EGE
        sal_resized = F.interpolate(sal[:, :1, :, :], size=(512, 512), mode='bilinear', align_corners=False)

        # 调整 sal_sig 的尺寸 light
        sal_sig_resized = F.interpolate(sal_sig[:, :1, :, :], size=(512, 512), mode='bilinear', align_corners=False)
        # sal = sal.view(1, 1, 1024, 1024) # 改变形状
        # sal_sig = sal_sig.view(1, 1, 1024, 1024)
        loss = CE(sal, gts) + IOU(sal_sig, gts)   # diyiban!!!
        # loss = CE(sal, gts) + IOU(sal_sig, gts)
        # loss = CE(sal, gts)
        # loss.backward()
        loss.backward(retain_graph=True)     #!!!!!!!RCM
        optimizer.step()



        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        total_loss += loss.item()
        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data))
    avg_loss = total_loss / total_step
    print(f'Epoch [{epoch}/{opt.epoch}], Average Loss: {avg_loss:.4f}')  # 打印平均损失

    save_path = 'models/LoveDA/RollingUNet/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'RollingUNet.pth' + '.%d' % epoch, _use_new_zipfile_serialization=False)

if  __name__ == '__main__':
    print("Let's go!")
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
