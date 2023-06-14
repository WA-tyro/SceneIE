import sys
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np

# from vit_pytorch import ViT
import torch.nn.functional as F
from loadData import loadeData, DataLoade
# from VITmodel import vit_large_patch16_224
# import torchsummary
from losses import loss, loss_SH, loss_render
from sh_render import SHmap, batch_size, device, shrender
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.utils.tensorboard import SummaryWriter

# def cal_RMSE(pred, gt):
#     # mse_loss = torch.nn.MSELoss
#     # rmse = torch.sqrt(mse_loss(pred, gt))
#     return torch.sqrt(torch.mean((pred - gt) ** 2))


# def cal_DSSIM(pred, gt):
#     return 1 - ssim(pred, gt, data_range=1, size_average=True)

coeff = np.loadtxt('coeff.txt')
continueTrain = False

def Focal(x):
    x = x.cpu().detach().numpy()
    h = x.shape[1]
    w = x.shape[2]
    steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
    steradian = np.sin(steradian / h * np.pi)
    steradian = np.tile(steradian.transpose(), (w, 1))
    steradian = steradian.transpose()
    steradian = steradian[..., np.newaxis]
    x = steradian * x
    x_intensity = 0.3 * x[..., 0] + 0.59 * x[..., 1] + 0.11 * x[..., 2]
    max_intensity_ind = np.unravel_index(np.argmax(x_intensity, axis=None), x_intensity.shape)
    max_intensity = x_intensity[max_intensity_ind]
    map = x_intensity > (max_intensity * 0.05)
    light = x * map[:,:,:,np.newaxis]
    light = torch.from_numpy(light)
    return light



def caculateTopk(x):
    k = int(len(x[0, :, :, :].flatten()) * 0.1)
    temp = torch.zeros((x.shape[0], k))
    for i in range(x.shape[0]):
        t = x[i, :, :, :].flatten()
        topk_vals, topk_idxs = t.topk(k)
        temp[i, :] = topk_vals
        return temp


def load_shamp(dir):
    # shmap, mask = SHmap(dir)
    shmap = SHmap(dir).to(device)
    return shmap


def creat_dataloader(trainList_path):
    img_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    loade_data = DataLoade(
        train_list=trainList_path, transform=img_transform
    )
    # data loader 数据载入
    data = DataLoader(
        dataset=loade_data, batch_size=batch_size, shuffle=True, num_workers=1
    )
    total_batch = data.__len__()
    return data, total_batch



def reconstruction(coeff, shcoeff, tag):
    batch = shcoeff.shape[0]
    coeff = torch.from_numpy(coeff)
    coeff = coeff.unsqueeze(0).to(device)
    if (tag == 'pred'):
        shcoeff = torch.reshape(shcoeff, (batch, 3, 16))
    h = 128
    w = 256
    # shcoeff = shcoeff * 255.0
    r_coeff = shcoeff[:, 0, :].unsqueeze(1)
    g_coeff = shcoeff[:, 1, :].unsqueeze(1)
    b_coeff = shcoeff[:, 2, :].unsqueeze(1)
    out_r = torch.sum(coeff * r_coeff, dim=2)
    out_g = torch.sum(coeff * g_coeff, dim=2)
    out_b = torch.sum(coeff * b_coeff, dim=2)

    sh_img = torch.zeros((batch, 128, 256, 3))
    sh_img[:, :, :, 0] = torch.reshape(out_r, (batch, w, h)).T.permute(2, 0, 1)
    sh_img[:, :, :, 1] = torch.reshape(out_g, (batch, w, h)).T.permute(2, 0, 1)
    sh_img[:, :, :, 2] = torch.reshape(out_b, (batch, w, h)).T.permute(2, 0, 1)
    # sh_img = torch.clamp(sh_img, 0, 255).to(torch.uint8)
    return sh_img


def initialize_model(net, optimizer, continue_train, weights_path='./weights/latest.pth', ):
    if continue_train:
        checkpoint = torch.load(weights_path, map_location=device)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        checkpoint = torch.load(weights_path, map_location=device)['model']
        net.load_state_dict(checkpoint, strict=False)
        print(net.load_state_dict(checkpoint, strict=False))

def train_one_epoch(model, optimizer, data_loader, device, epoch, shmap, training_examples, iter, savdir):
    Loss_sum = torch.zeros(1).to(device)
    model.train()
    # linear.train()
    # data_loader = tqdm(data_loader, file=sys.stdout)
    # for i, imgsAndSH in enumerate(data_loader, start=totle_iter):
    for i, imgsAndSH in enumerate(data_loader, start=iter):

        print("epoch: {}  [{}/{}]".format(epoch, i + 1, training_examples), end='  ')
        optimizer.zero_grad()
        predict = model(imgsAndSH[0].to(device))
        sh_gt = imgsAndSH[1].to(device)

        # 恢复图像
        shimagepred = reconstruction(coeff, predict, 'pred')
        shimagegt = reconstruction(coeff, sh_gt, 'gt')

        # # 计算FOCAL前5%亮的像素
        # imagepred_focal = Focal(shimagepred)
        # imagegt_focal = Focal(shimagegt)

        # 先恢复图像再计算损失
        Loss = F.mse_loss(shimagepred, shimagegt)
        Loss_shimage = F.mse_loss(shimagepred, shimagegt)
        # Loss_Focal = F.mse_loss(imagepred_focal, imagegt_focal)

        # # 直接计算sh系数的损失
        # Loss = F.mse_loss(sh_gt.float(), torch.reshape(predict, (predict.shape[0], 3, 16)))
        # Loss = loss(predict, sh_gt, shmap, 0.5)

        # Loss = loss_SH(predict, sh_gt)

        print('lr: {}'.format(optimizer.param_groups[0]["lr"]), end="  ")
        Loss.backward()
        optimizer.step()
        # scheduler.step()
        Loss_sum += Loss.item()

        print("loss_sh:{}, loss_shimage:{}, loss_render:{},loss_merage:{}".format(
            loss_SH(predict, sh_gt).cpu().detach().numpy(),
            Loss_shimage.cpu().detach().numpy(),
            loss(predict, sh_gt, shmap, 0.8).cpu().detach().numpy(),
            # Loss_Focal.cpu().detach().numpy(),
            Loss.cpu().detach().numpy())
        )

        if (i+1) % 500 == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, (epoch - 1) * training_examples + i))
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'iter': i + 1
            }
            torch.save(checkpoint, savdir + "/latest.pth")
            print('Saved current weights at %s.' % "./weights/latest.pth")

    print('saving the latest model (epoch %d, total_steps %d)' % (epoch, epoch * training_examples))
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iter': training_examples
    }
    torch.save(checkpoint, savdir + "/latest.pth")
    print('Saved current weights at %s.' % "./weights/latest.pth")

    print("Epoch:{} average loss: ".format(epoch), Loss_sum / torch.tensor(training_examples))
    return Loss_sum / torch.tensor(training_examples)


def eval_one_epoch(model, data_loader, device, epoch, shmap, Eval_examples):
    print("Evaluating")
    model.eval()
    # linear.eval()
    Loss_sumEval = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for i, imgsAndSH in enumerate(data_loader):
            predict = model(imgsAndSH[0].to(device))
            sh_gt = imgsAndSH[1].type(torch.float32).to(device)

            # 恢复图像
            shimagepred = reconstruction(coeff, predict, 'pred')
            shimagegt = reconstruction(coeff, sh_gt, 'gt')

            # # 计算FOCAL前10%亮像素
            # imagepred_top = caculateTopk(shimagepred)
            # imagegt_top = caculateTopk(shimagegt)


            # 先恢复图像再计算损失
            Loss = F.mse_loss(shimagepred, shimagegt)

            # # 直接计算sh系数的损失
            # Loss = F.mse_loss(sh_gt, torch.reshape(predict, (predict.shape[0], 3, 16)))
            # Loss = loss(predict, sh_gt, shmap, 0.5)
            # Loss = loss_SH(predict, sh_gt)
            Loss_sumEval += Loss.item()
            # print("loss: {}".format(Loss))
    print("Epoch:{} Evaluating average loss: ".format(epoch), Loss_sumEval / torch.tensor(Eval_examples))
    return Loss_sumEval / torch.tensor(Eval_examples)
