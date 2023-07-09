import os
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt


class Caculate:
    def __init__(self):
        self.rmse = 0.0
        self.ssim = 0.0
        self.si_rmse = 0.0
        self.angular = 0.0
        self.count = 0

    def read_iamge(self, image_path):
        image = Image.open(image_path).convert('RGB')
        i = plt.imread(image_path)
        return transforms.ToTensor()(image), i

    def calculate_rmse(self, image1, image2):
        # diff = image1 - image2
        # mse = (diff ** 2).mean()
        # rmse = mse ** 0.5
        return torch.sqrt(torch.nn.MSELoss()(image1, image2))

    def scale_invariant_loss(self, image1, image2, reduction="mean"):
        """
        si_RMSE
        outs: N ( x C) x H x W
        targets: N ( x C) x H x W
        reduction: ...
        """
        outs = image1.flatten(start_dim=1)
        targets = image2.flatten(start_dim=1)
        alpha = (targets - outs).mean(dim=1, keepdim=True)
        return F.mse_loss(outs + alpha, targets, reduction=reduction)

    def calculate_dssim(self, image1, image2):
        Ssim = ssim(image1.permute(1, 2, 0).detach().cpu().numpy(), image2.permute(1, 2, 0).detach().cpu().numpy(),
                    multichannel=True)
        return Ssim

    def get_metric(self):
        return {
            'rmse': self.rmse / self.count,
            'dssim': 1 - self.ssim / self.count,
            'si-rmse': self.si_rmse / self.count,
            'angular': self.angular / self.count
        }

    def calculate_angular(self, pred, gt):
        h = pred.shape[0]
        w = pred.shape[1]
        mask = np.zeros((h, w))
        mask[pred[:, :, 3] > 0] = 1
        num_pixel = mask.sum()

        cos = np.sum(gt[:, :, :3] * pred[:, :, :3], axis=2) / (
                np.linalg.norm(gt[:, :, :3], axis=2) * np.linalg.norm(pred[:, :, :3], axis=2))
        cos = np.clip(cos, 0, 1)
        angular = np.nan_to_num(np.multiply(np.arccos(cos), mask))

        res = np.degrees(angular.sum() / num_pixel)

        return res

    def forward(self, path_gt, path_pred):
        image_gt, i_g = self.read_iamge(path_gt)
        image_pred, i_p = self.read_iamge(path_pred)
        rmse = self.calculate_rmse(image_gt, image_pred)
        angular = self.calculate_angular(i_p, i_g)
        self.rmse += rmse
        si_rmse = self.scale_invariant_loss(image_gt, image_pred)
        self.si_rmse += si_rmse

        ssim = self.calculate_dssim(image_gt, image_pred)
        self.ssim += ssim

        self.angular += angular

        print(path_gt)
        print('RMSE: %.4f, DSSIM: %.4f, Si-rmse: %.4f, angular: %.4f' % (rmse, 1 - ssim, si_rmse, angular))
        self.count += 1


caculate = Caculate()
gt_path = '/media/wangao/DataDisk/Study/work/Scence_Illumination/GAN_work/gt/WrapedGT/'
pre_path = '/media/wangao/DataDisk/Study/Data/shlight/results/test1800/original/rendered/'
gt_names = [f for f in os.listdir(gt_path) if f.endswith('png')]
pred_names = [f for f in os.listdir(pre_path) if f.endswith('png')]

for pred_name in pred_names:
    path_g = gt_path + pred_name
    if pred_name in gt_names:
        path_p = pre_path + pred_name
        caculate.forward(path_g, path_p)
        metric = caculate.get_metric()
        print('Mean RMSE: %.4f, Mean DSSIM: %.4f, Mean si-rmse: %.4f, , Mean angular: %.4f\n' % (
            metric['rmse'], metric['dssim'], metric['si-rmse'], metric['angular']))
