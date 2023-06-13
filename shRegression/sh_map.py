import numpy as np
import cv2 as cv
import torch

batch_size = 32

#
def SHmap(shMap_dir):
    shmap = np.load(shMap_dir)
    # if bunny.npy
    shmap = shmap.reshape((256, 256, 16, 1))
    mask = np.abs(shmap).sum(axis=-2).reshape((256, 256, 1))
    mask = mask.astype(np.bool).astype(np.uint8)
    # # if horse.npy
    # shmap = shmap.reshape((360, 480, 16, 1))
    # mask = np.abs(shmap).sum(axis=-2).reshape((360, 480, 1))
    return shmap, mask


def render(sh, shmap):
    sh = sh.reshape((1, 1, 16, 3))
    image = np.sum(sh * shmap, axis=-2)
    image = np.clip(image, 0, 1)
    image = image ** (1 / 2.2)

    return image


def visualize(sh_gt, pred, mask):
    pred = pred * mask + sh_gt * (1 - mask)
    cv.imshow('gt', sh_gt[:, :, (2, 1, 0)])
    cv.imshow('pd', pred[:, :, (2, 1, 0)])
    assert cv.waitKey(0) != 27



#
# import numpy as np
#
# a = np.ones((2, 3, 5, 4))
# print(a)
# print(a.sum(axis=3))
