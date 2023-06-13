import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import time
import os
import imageio
imageio.plugins.freeimage.download()
import util

handle = util.PanoramaHandler()
tone = util.TonemapHDR()

bs_dir = '/home/wangao/PycharmProjects/Illumination_Estimation/EMLight/warped_Laval_Indoor_HDR_Dataset/'
exr_dir = bs_dir + 'warpedHDROutputs/'
sv_dir = bs_dir + 'warpedHDROutputs/bmp/'
nms = os.listdir(exr_dir)
# nms = nms[:100]

i = 0
for nm in nms:
    if nm.endswith('.exr'):
        exr_path = exr_dir + nm
        exr = handle.read_exr(exr_path)

        im = tone(exr, True).clip(0, 1)
        # plt.imsave(sv_dir + nm.replace('exr', 'bmp'), im)
        # im = tone(exr, True)
        # 保存数值在（0，1）之间
        # plt.imsave(sv_dir + nm.replace('exr', 'bmp'),im)
        # emlight是这么写的，因求SH matlab和python读取不一样，故改写
        im = Image.fromarray((im*255.0).astype('uint8'))
        sv_path = sv_dir + nm.replace('exr', 'jpg')
        im.save(sv_path)
        i += 1
        print (i)