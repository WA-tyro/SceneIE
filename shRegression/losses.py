from sh_render import SHmap, shrender, batch_size
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def reconstruction(coeff, shcoeff):
    h = 128
    w = 256
    # shcoeff = shcoeff * 255.0
    r_coeff = shcoeff[0,:]
    g_coeff = shcoeff[1,:]
    b_coeff = shcoeff[2,:]
    out_r = np.sum(coeff * r_coeff, axis=1)

    out_g = np.sum(coeff * g_coeff, axis=1)

    out_b = np.sum(coeff * b_coeff, axis=1)

    sh_img = np.zeros((128,256,3))
    sh_img[:, :, 0] = np.reshape(out_r, (h, w), order='F')
    sh_img[:, :, 1] = np.reshape(out_g, (h, w), order='F')
    sh_img[:, :, 2] = np.reshape(out_b, (h, w), order='F')
    sh_img = sh_img.clip(0, 255).astype(np.uint8)
    return sh_img

def loss_render(sh_pred, sh_gt, shmap):
    pred = shrender(sh_pred, shmap)
    gt = shrender(sh_gt, shmap, 'gt')
    # result = torch.clip(pred, 0, 1)
    # result = result ** (1 / 2.2)
    # result = result[0,:,:,:]
    # result = result.cpu().detach().numpy()
    #
    # gt = torch.clip(gt, 0, 1)
    # gt = gt ** (1 / 2.2)
    # gt = gt[0, :, :, :]
    # gt_ = gt.cpu().detach().numpy()
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(result)
    # plt.subplot(2, 2, 2)
    # plt.imshow(gt_)
    # plt.show()


    return torch.mean(
        torch.sum(torch.square(pred - gt), dim=[1, 2]) / torch.count_nonzero(torch.sum(shmap, [-1])))


# def loss_SH(sh_pred, sh_gt):
#     return F.mse_loss(sh_pred, sh_gt)
#     # return torch.mean(torch.square(sh_pred-sh_gt))

def loss_SH(sh_pred, sh_gt):
    sh_pred = torch.reshape(sh_pred, (sh_pred.shape[0], 3, 16))
    return F.mse_loss(sh_pred, sh_gt)
    # return torch.mean(torch.square(sh_pred-sh_gt))


def loss_mse_weighted(pred, gt):
    pred = torch.reshape(pred, (pred.shape[0], 3, 16))
    weights = torch.tensor([1/12]*3 + [1/36]*9 + [1/60]*15 + [1/84]*21, device=pred.device)
    return torch.mean(torch.sum(torch.multiply(torch.square(pred - gt), weights), dim=-1))


def loss(sh_pred, sh_gt, shmap, ratio):
    # sh_pred = torch.reshape(sh_pred,[-1,16,3])
    # shloss = loss_mse_weighted(sh_pred, sh_gt)
    shloss = loss_SH(sh_pred, sh_gt)
    rdloss = loss_render(sh_pred, sh_gt, shmap)
    return torch.mul(shloss, ratio) + torch.mul(rdloss, 1 - ratio)

# img1 = np.array([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
#                  [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]])
# img2 = np.array([[[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]], [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]],
#                  [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]]])
# img1 = torch.tensor(img1)
# img2 = torch.tensor(img2)
# a = torch.sum(torch.square(img2 - img1),axis=[1,2])
# pass


# shmap, mask = SHmap('../Data/shmap/bunny.npy')
#
# v = ViT(
#     image_size=224,
#     patch_size=32,
#     num_classes=1000,
#     dim=1024,
#     depth=6,
#     heads=16,
#     mlp_dim=2048,
#     dropout=0.1,
#     emb_dropout=0.1
# )
# batch_size = 8
#
# img = torch.randn(batch_size, 3, 224, 224)
# preds = v(img)
# preds = preds.reshape(batch_size, 1, 16, 3)
# sh = open('../Data/sh/0001/000.txt').readlines()
# sh_gt = []
# for line in sh:
#     line = line.strip('\n').split(' ')
#     for sh_coff in line:
#         sh_gt.append(float(sh_coff))
# sh_gt = np.array(sh_gt)
# sh_gt = sh_gt.reshape(batch_size, 1, 16, 3)
# sh_gt = torch.tensor(sh_gt)
# shmap = torch.tensor(shmap)
# loss = loss_render(preds, sh_gt, shmap)
# pass
