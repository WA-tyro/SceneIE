import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import cv2

base_dir = '/home/wangao/PycharmProjects/code/'


class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img.astype('float32')


tone = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)


def read_hdr(hdr_path):
    hdr_img = cv2.imread(hdr_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    hdr_img = hdr_img[..., ::-1]
    return hdr_img


class loadeData(Dataset):
    def __init__(self, train_list, transform=None, ):
        self.train_list = train_list
        self.transform = transform
        f = open(self.train_list, 'r')
        self.input = f.readlines()

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_index = self.input[index]
        # image_input1 = Image.open(input_index.split(" ")[0])
        # sh = torch.tensor(np.loadtxt(input_index.split(" ")[1].strip('\n'),delimiter=' ',dtype='float32').ravel())

        image_input1 = Image.open(input_index.split(" ")[0].replace('../', base_dir))
        image_input2 = Image.open(input_index.split(" ")[1].replace('../', base_dir))
        sh = torch.tensor(np.loadtxt(input_index.split(" ")[2].strip('\n').replace('../', base_dir), delimiter=' ',
                                     dtype='float32').ravel())

        # sh_gt = sh.ravel('F')
        if self.transform:
            image_input1 = self.transform(image_input1)
            image_input2 = self.transform(image_input2)
        return image_input1, image_input2, sh


class DataLoade(Dataset):
    def __init__(self, train_list, transform=None, ):
        self.train_list = train_list
        self.transform = transform
        f = open(self.train_list, 'r')
        self.input = f.readlines()

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_index = self.input[index].strip('\n')
        sh = np.loadtxt(input_index.split("*")[1])
        sh = torch.FloatTensor(sh)
        # 使用IndoorHDR的时候按行展开,使用code的数据集时候,需要按列展开sh.ravel('F')
        # sh_gt = sh.T.ravel()
        image_path = input_index.split("*")[0]
        hdr = read_hdr(image_path)
        image_input = (torch.from_numpy(tone(hdr))).permute(2, 0, 1)
        # image_input = Image.open(image_path)
        # image_new = Image.new("RGB", (256, 256), (0, 0, 0))
        # image_new.paste(image_input)
        # sh_gt = sh_gt.type(torch.float32)
        # sh_gt = torch.tensor(sh_gt)
        if self.transform:
            image_input = self.transform(image_input)
        return image_input, sh
