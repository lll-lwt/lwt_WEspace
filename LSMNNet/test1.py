import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc
import imageio
import time
import ml_collections

# from model.GeleNet_models import GeleNet
from data import test_dataset
from model.H_vmunet import H_vmunet
from model.LightMUNet import LightMUNet
import numpy as np
from PIL import Image
# from model.RS3Mamba import RS3Mamba
# from model.RS3att import RS3Mamba
# from model.SpikingRS314 import RS3Mamba
# from model.back1 import RS3Mamba
# from model.egeunet import EGEUNet
from model.UNet import UNet
from model.DeeplabV3Plus import DeepLab
# from model.Swin_Unet import SwinUnet
from model.RollingUnet import Rolling_Unet_L
from model.RS3att import RS3Mamba
from model.SCTransNet import SCTransNet
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
opt = parser.parse_args()

dataset_path = '/home/lwt/桌面/Dataset/flair/RGB2/'
dataGT_path = '/home/lwt/桌面/Dataset/flair/label2/'





def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1

    # ********** useless **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config
config_vit = get_CTranS_config()
model = SCTransNet(config_vit, mode='test', deepsuper=True)
# model = GeleNet()
# model = H_vmunet()
# model = LightMUNet()
# model = Rolling_Unet_L(1)
# model = EGEUNet()
# model = UNet()
# model = DeepLab(1)
# model = SwinUnet()
model.load_state_dict(torch.load('/home/lwt/桌面/GeleNet-ori/models/BBBBB/SCTransNet/SCTransNet.pth.84'))
# model.cpu()
model.cuda()
model.eval()






test_datasets = ['SCT']
#test_datasets = ['EORSSD','ORSSD','ors-4199']

for dataset in test_datasets:
    save_path = '/home/lwt/桌面/results/BBBBB/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path
    print(dataset)
    gt_root = dataGT_path
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, sal_sig = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        # res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # res_uint8 = (res * 255).astype(np.uint8)   #xpl
        # imageio.imsave(save_path + name, res_uint8)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # 应用阈值处理将图像转换为二值图像
        threshold = 0.6  # 设置阈值
        binary_mask = (res > threshold).astype(np.uint8)
        # binary_mask = (res > threshold).max(axis=0).astype(np.uint8)   # LightMUNet格式转化
        # 保存二值图像，使用输入图片的名称
        imageio.imsave(save_path + name, binary_mask * 255)
        # imageio.imsave(save_path+name, res)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('FPS {:.5f}'.format(test_loader.size / time_sum))

    # for i in range(test_loader.size):
    #     image, gt, name = test_loader.load_data()
    #     gt = np.asarray(gt, np.float32)
    #     gt /= (gt.max() + 1e-8)
    #     image = image.cuda()
    #     time_start = time.time()
    #     res, sal_sig = model(image)
    #     time_end = time.time()
    #     time_sum = time_sum + (time_end - time_start)
    #     res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    #     res = res.sigmoid().data.cpu().numpy().squeeze()
    #     res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    #     res_uint8 = (res * 255).astype(np.uint8)  # xpl
    #     imageio.imsave(save_path + name, res_uint8)
    # # imageio.imsave(save_path+name, res)
    #     if i == test_loader.size - 1:
    #         print('Running time {:.5f}'.format(time_sum / test_loader.size))
    #         print('FPS {:.5f}'.format(test_loader.size / time_sum))