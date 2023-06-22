import os

import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import matplotlib.pyplot as plt


class Caculate:
    def __init__(self):
        self.rmse = 0.0
        self.ssim = 0.0
        self.count = 0

    def read_iamge(self, image_path):
        image = imread(image_path)
        return image

    def calculate_rmse(self,image1, image2):
        diff = image1 - image2
        mse = (diff ** 2).mean()
        rmse = mse ** 0.5
        return rmse

    def calculate_dssim(self,image1, image2):
        dssim = ssim(image1, image2, multichannel=True)
        return dssim

    def get_metric(self):
        return {
            'rmse': self.rmse / self.count,
            'dssim': 1- self.ssim / self.count,
        }

    def forward(self, path_gt, path_pred):
        image_gt = self.read_iamge(path_gt)
        image_pred = self.read_iamge(path_pred)
        rmse = self.calculate_rmse(image_gt, image_pred)
        self.rmse += rmse

        ssim = self.calculate_dssim(image_gt, image_pred)
        self.ssim += ssim

        print(path_gt)
        print('RMSE: %.4f, DSSIM: %.4f' % (rmse, 1 - ssim))
        self.count += 1


caculate = Caculate()
gt_path = '/home/wangao/PycharmProjects/Illumination_Estimation/shlight/GenProjector/results/NoneShadow/GT/'
pre_path = '/home/wangao/PycharmProjects/Illumination_Estimation/shlight/GenProjector/results/NoneShadow/rendered/'
gt_names = [f for f in os.listdir(gt_path) if f.endswith('png')]
pred_names = [f for f in os.listdir(pre_path) if f.endswith('png')]

for pred_name in pred_names:
    path_g = gt_path + pred_name
    if pred_name in gt_names:
        path_p = pre_path + pred_name
        caculate.forward(path_g,path_p)
        metric = caculate.get_metric()
        print('Mean RMSE: %.4f, Mean DSSIM: %.4f\n' % (metric['rmse'], metric['dssim']))
