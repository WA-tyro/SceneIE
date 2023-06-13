import numpy as np
import torch

batch_size = 8
device = 'cuda'


def selectSHMap(concated_shmap):
    index = torch.randint(10, (1,))
    offset = 256 * torch.min(index, torch.tensor([3], dtype=torch.int32))
    begin = torch.cat((offset, torch.tensor([0, 0], dtype=torch.int32)))
    selector = concated_shmap[begin[0]:begin[0] + 256, begin[1]:begin[1] + 512, begin[2]:begin[2] + 16]
    return selector


def SHmap(shMap_dir):
    shmap = np.load(shMap_dir)
    shmap = selectSHMap(shmap)
    # # if bunny.npy
    # shmap1 = shmap.reshape((256, 256, 16))
    # shmap = shmap.reshape((256, 256, 16, 1))
    # mask = np.abs(shmap1).sum(axis=-1).reshape((256, 256, 1))
    # mask = mask.astype(np.bool).astype(np.uint8)

    # # if horse.npy
    # shmap1 = shmap.reshape((360, 480, 16))
    # shmap = shmap.reshape((360, 480, 16, 1))
    # mask = np.abs(shmap1).sum(axis=-1).reshape((360, 480, 1))
    # mask = mask.astype(np.bool).astype(np.uint8)

    # # if concated.npy
    # shmap1 = shmap.reshape((1024, 512, 16))
    # shmap = shmap.reshape((1024, 512, 16, 1))
    # mask = np.abs(shmap1).sum(axis=-1).reshape((1024, 512, 1))
    # mask = mask.astype(np.bool).astype(np.uint8)

    # if envmap.npy
    # shmap1 = shmap.reshape((256, 512, 16))
    shmap = shmap.reshape((256, 512, 16, 1))
    # mask = np.abs(shmap1).sum(axis=-1).reshape((256, 512, 1))
    # mask = mask.astype(np.bool).astype(np.uint8)
    return torch.tensor(shmap)


def render(sh, shmap):
    sh = sh.reshape((batch_size, 1, 16, 3))
    image = torch.sum(sh * shmap, dim=-2)

    return image


def shrender(sh, shmap, tag='pred'):
    # sh :   tensor[batch_size,48]
    # shmap: tensor[height,width,16]
    shmap = torch.unsqueeze(shmap, 0)  # shmap : tensor[1,height,width,16]
    if tag == 'gt':
        sh = torch.reshape(sh.T.permute(2, 0, 1), (sh.T.permute(2, 0, 1).shape[0], 48))
    sh = torch.reshape(sh, [-1, 1, 1, 16, 3])  # [batch_size,1,1,16,3]
    result = torch.sum(sh * shmap, dim=3)  # [batch_size, height, width, 3]
    return result


# if __name__ == "__main__":
#     sh = np.loadtxt('../SH/GeneratorGroundTruth/SHGT/9C4A0001-others-00-2.14020-1.00216.txt', delimiter=' ')
#
#     sh = sh.ravel('F')
#     # sh = sh.T.flatten()
#     sh = sh.reshape(1, 1, 16, 3)
#     sh = torch.from_numpy(sh)
#     sh_gt = sh.unsqueeze(0)
#
#     # print(SHmap('../Data/shmap/bunny.npy'))
#     shmap, mask = SHmap('../Data/shmap/concated.npy')
#     # shmap = np.load('../Data/shmap/bunny.npy')
#     # shmap = torch.tensor(shmap)
#     # shmap = torch.reshape(shmap, (256, 256, 16))
#     # shmap = torch.unsqueeze(shmap,0)
#     # shmap = torch.unsqueeze(shmap,-1)
#     # image = shrender(sh_gt, shmap)
#
#     # image = np.sum(sh * shmap, axis=-2)
#     image = torch.sum(sh_gt * shmap, dim=-2)
#     # image = np.clip(image, 0, 1)
#     image = torch.clip(image, 0, 1)
#     image = image ** (1 / 2.2)
#
#     # image = shrender(sh_gt, shmap,'gt')
#     # image = torch.clip(image, 0, 1)
#     # image = image ** (1 / 2.2)
#     image = image.squeeze(0).cpu().detach().numpy()
#     plt.imshow(image)
#     plt.show()
#     pass
