import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from sh_render import device
import utils
from torch.utils.tensorboard import SummaryWriter
import models_vit
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def read_hdr(hdr_path):
    hdr_img = cv2.imread(hdr_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    hdr_img = hdr_img[..., ::-1]
    return hdr_img

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
sv_dir = './predictSHimage/'
coeff = np.loadtxt('coeff.txt')
model = models_vit.__dict__['vit_base_patch16'](
        img_size= 256,
        num_classes=48,
        drop_path_rate=0.1,
        global_pool=False,
    )

model.load_state_dict(torch.load("./runs/256x256InputVit/weights/model-73.pth"),strict=False)
print(model.load_state_dict(torch.load("./runs/256x256InputVit/weights/model-73.pth"),strict=False))
model.to(device)

input_path =  '/media/wangao/DataDisk/Study/Dataset/Laval Indoor HDR Daset/train/HDRInput/'
input_names = [f for f in os.listdir(input_path) if f.endswith('.exr')]
for name in input_names:
    hdr_path = input_path + name
    hdr = read_hdr(hdr_path)
    image = (torch.from_numpy(tone(hdr))).permute(2, 0, 1).unsqueeze(0).to(device)
    predict = model(image)
    gt_path = '/media/wangao/DataDisk/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/SHGT/train/' + name.replace('.exr','.txt')
    gt = torch.FloatTensor(np.loadtxt(gt_path)).unsqueeze(0).to(device)
    shimagepred = utils.reconstruction(coeff, predict, 'pred')
    shimagegt = utils.reconstruction(coeff, gt, 'gt')
    shimagepred = shimagepred.squeeze(0).detach().cpu().numpy()
    shimagegt = shimagegt.squeeze(0).detach().cpu().numpy()
    result = np.hstack((shimagegt,shimagepred))
    plt.imsave(sv_dir+'result' + '/{}'.format(name.replace('.exr','.jpg')),tone(result[0]))
    # result = Image.fromarray(result[0])
    # result.save(sv_dir+'result' + '/{}'.format(name.replace('.exr','.jpg')))
    parametric_lights = {'sh_image': shimagepred}
    with open((sv_dir + os.path.basename(hdr_path).replace('exr', 'pickle')), 'wb') as handle:
        pickle.dump(parametric_lights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(sv_dir+'result' + '/{}'.format(name.replace('.exr','.jpg')))
