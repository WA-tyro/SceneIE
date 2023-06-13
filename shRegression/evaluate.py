from utils import load_shamp
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import torch
from skimage.metrics import structural_similarity as compare_ssim
import torchvision.transforms as transforms
import models_vit
from loadData import tone, read_hdr

device = 'cuda'


def gen_eval_txt():
    with open('./Data/test.txt') as f:
        content = f.readlines()

    with open('./Data/eval.txt', 'w') as f:
        for line in content:
            f.write(line.replace('/sh/', '/gt/').replace('.txt', '_horse.jpg'))


class Evaluator:
    def __init__(self):
        shmap = torch.tensor(np.load('./Data/shmap/horse.npy')).to(device)
        self.shmap = shmap.reshape((360, 480, 16, 1))
        mask = torch.abs(self.shmap).sum(dim=-2).reshape((360, 480, 1))
        self.mask = mask.bool().to(torch.uint8)

        # self.mask_count = np.sum(self.mask) # 30264
        self.mask_count = 29505

        self.rmse = 0.0
        self.ssim = 0.0
        self.count = 0

    def _get_image_gt(self, filename):
        image = imread(filename)
        image = torch.tensor(image / 255.0)
        return image.to(device)

    # 直接使用sh_gt渲染，然后再计算ssim

    def _get_gt(self, filename):
        sh_gt = np.loadtxt(filename)
        sh_gt = torch.from_numpy(sh_gt).to('cuda')
        image = self._render(sh_gt)
        return image.to(device)

    def _render(self, sh):
        sh = sh.reshape(3, 16).T

        sh = sh.reshape((1, 1, 16, 3))
        image = torch.sum(sh * self.shmap, dim=-2)
        # image = image.cpu().detach().numpy()
        # if np.all(image == 0.0):
        #     print("True")
        # else:
        #     nonzero_coords = np.where(image != 0.0)
        #     for x, y in zip(nonzero_coords[0], nonzero_coords[1]):
        #         print(f"Element at ({x}, {y}) is {image[x, y]}")
        #     print("False")
        image = torch.clip(image, 0, 1)
        image = image ** (1 / 2.2)

        return image

    def _compute_rmse(self, gt, pred):
        # RMSE on unmasked images
        # return np.sqrt(np.mean(np.square(gt - pred)))

        # RMSE on masked images
        channel_mean = torch.mean(torch.square(gt - pred), dim=-1)
        return torch.sqrt(torch.sum(channel_mean) / self.mask_count).item()

    def _compute_ssim(self, gt, pred):
        return compare_ssim(gt[30:330, 40:410, :].cpu().detach().numpy(),
                            pred[30:330, 40:410, :].cpu().detach().numpy(), multichannel=True)

    def forward(self, sh, gt_filename):
        image_gt = self._get_gt(gt_filename)
        image_pred = self._render(sh)

        # image = image_pred.cpu().detach().numpy()
        # image_gt = image_gt.cpu().detach().numpy()
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(image)
        # plt.subplot(2, 2, 2)
        # plt.imshow(image_gt)
        # plt.show()

        rmse = self._compute_rmse(image_gt * self.mask, image_pred)
        self.rmse += rmse

        ssim = self._compute_ssim(image_gt * self.mask, image_pred)
        self.ssim += ssim

        print(gt_filename)
        print('RMSE: %.4f, DSSIM: %.4f' % (rmse, 1 - ssim))
        self.count += 1

        # pred = image_pred * self.mask + image_gt * (1 - self.mask)
        # pred = pred * 255
        # pred = pred.cpu().detach().numpy()
        # cv.imshow('',pred[:, :, (2, 1, 0)])
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # cv.imwrite('/media/wangao/DataDisk/Study/PycharmProject/VIT/vit-pytorch/Data/result_photo/' + gt_filename[-18:-14] + '_' + gt_filename[-13:],pred[:, :, (2, 1, 0)])

    def get_metric(self):
        return {
            'rmse': self.rmse / self.count,
            'dssim': 1 - self.ssim / self.count,
        }


if __name__ == '__main__':
    gen_eval_txt()

    model = models_vit.__dict__['vit_base_patch16'](
        img_size=256,
        num_classes=48,
        drop_path_rate=0.1,
        global_pool=False,
    )
    model.to(device)
    # linear.to(device)
    model.eval()
    # linear.eval()
    evaluator = Evaluator()

    checkpoint = torch.load("./weights/model-65.pth", map_location=device)
    model.load_state_dict(checkpoint)
    # linear.load_state_dict(checkpoint['linear'])
    print("模型加载成功，准备预测啦")

    with open('./Data/list/test.txt', 'r') as f:
        # with open('/home/wangao/PycharmProjects/code/data/list/eval.txt', 'r') as f:
        lines = f.readlines()

    with torch.no_grad():
        for line in lines:
            input = torch.FloatTensor(tone(read_hdr(line.split("*")[0])))
            input = input.permute(2, 0, 1)
            input = input.unsqueeze(0).to(device)
            # gt = imread(line.split(" ")[1].strip('\n'))
            # input = transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3)(input)
            pred = model(input)

            # input = torch.FloatTensor(plt.imread('/home/wangao/PycharmProjects/code/data/im/0032/000_0.jpg'))
            # input = input.cpu().detach().numpy()

            # image_input1 = torch.FloatTensor(imread(line.split(" ")[0].replace('../', base_dir)))
            # image_input2 = torch.FloatTensor(imread(line.split(" ")[1].replace('../', base_dir)))
            # # gt = imread(line.split(" ")[2].strip('\n').replace('../', base_dir))
            # image_input1 = image_input1.permute(2, 1, 0)
            # image_input2 = image_input2.permute(2, 1, 0)
            # image_input1 = image_input1.unsqueeze(0).to(device)
            # image_input2 = image_input2.unsqueeze(0).to(device)
            # feature1 = model(image_input1)
            # feature2 = model(image_input2 )
            # feature_fusion = torch.concat([feature1, feature2],dim=-1)
            # pred = linear(feature_fusion)

            evaluator.forward(pred, line.split("*")[1].strip('\n'))
            metric = evaluator.get_metric()
            print('Mean RMSE: %.4f, Mean DSSIM: %.4f\n' % (metric['rmse'], metric['dssim']))
