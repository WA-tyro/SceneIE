import os
import time

import numpy as np
import torch
import torch.nn as nn
from sh_render import device
import utils
from torch.utils.tensorboard import SummaryWriter
import models_vit
if os.path.exists("./weights") is False:
    os.makedirs("./weights")

tb_writer = SummaryWriter()
shmap = utils.load_shamp('./Data/shmap/concated.npy')

continueTrain = utils.continueTrain
epochs = 100

# train_data, total_train = utils.creat_dataloader('/home/wangao/PycharmProjects/code/data/list/train.txt')
# Eval_data, total_eval = utils.creat_dataloader('/home/wangao/PycharmProjects/code/data/list/test.txt')
# train_data = utils.creat_dataloader('/media/wangao/DataDisk/Study/PycharmProject/VIT/vit-pytorch/SH/GeneratorGroundTruth/train.txt')
# Eval_data = utils.creat_dataloader('/media/wangao/DataDisk/Study/PycharmProject/VIT/vit-pytorch/SH/GeneratorGroundTruth/test.txt')
train_data, total_train = utils.creat_dataloader('./Data/list/train.txt')
Eval_data, total_eval = utils.creat_dataloader('./Data/list/test.txt')

# model = MobileNetV3_Small()
#
# model = MobileNetV3.MobileNetV3()

# linear = nn.Linear(2560, 48)
# linear.add_module('drop',nn.Sequential(nn.Dropout(p=0.2, inplace=True)))
# linear = Dense()

# model = ViT(
#     image_size=224,
#     patch_size=16,
#     num_classes=1000,
#     dim=1024,
#     depth=12,
#     heads=16,
#     mlp_dim=2048,
#     dropout=0.1,
#     emb_dropout=0.1
# )
model = models_vit.__dict__['vit_base_patch16'](
        img_size= 256,
        num_classes=48,
        drop_path_rate=0.1,
        global_pool=False,
    )

# model = AlexNet()

# model = vit_large_patch16_224(48)
# weights_dict = torch.load("../mae_pretrain_vit_base.pth", map_location=device)
# del_keys = ['head.weight', 'head.bias']
# for k in del_keys:
#     del weights_dict[k]
# print(model.load_state_dict(weights_dict, strict=False))
# model.load_state_dict(weights_dict, strict=False)
# # model.load_state_dict(torch.load("../vit_large_patch16_224.pth"))
# torchsummary.summary(model, (3, 224, 224), device=device)
model.to(device)
# linear.to(device)

# checkpoint_model = torch.load("../mae_pretrain_vit_base.pth", map_location=device)['model']
# state_dict = model.state_dict()
# for k in ['head.weight', 'head.bias']:
#     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#         print(f"Removing key {k} from pretrained checkpoint")
#         del checkpoint_model[k]
# print(model.load_state_dict(checkpoint_model, strict=False))
# model.load_state_dict(checkpoint_model, strict=False)


# optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, betas=(0.9, 0.99), eps=1e-8)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.9, last_epoch=-1)

if continueTrain:
    utils.initialize_model(model, optimizer, continueTrain)
    checkpoint = torch.load('./weights/latest.pth', map_location=device)
    first_epoch, iter = checkpoint['epoch'], checkpoint['iter']
    training_epochs = range(first_epoch, epochs+1)
else:
    # utils.initialize_model(model, optimizer, continueTrain, './mae_pretrain_vit_base.pth')
    training_epochs = range(1, epochs+1)
    iter = 0
    first_epoch = 1



for epoch in training_epochs:
    # print(model)
    print("training" + '\n' + "Epoch:{} 训练开始".format(epoch))
    # print(model.parameters())
    train_loss = utils.train_one_epoch(model, optimizer, train_data, device, epoch, shmap, total_train, iter)
    iter = 0
    eval_loss = utils.eval_one_epoch(model, Eval_data, device, first_epoch, shmap, total_eval)

    tags = ["train_loss", "val_loss", "learning_rate",'lr&loss']
    tb_writer.add_scalar(tags[0], train_loss, epoch)
    tb_writer.add_scalar(tags[1], eval_loss, epoch)
    tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
    tb_writer.add_scalar(tags[2], train_loss, optimizer.param_groups[0]["lr"])

    torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    # torch.save({
    #     'feature_Extract':model.state_dict(),
    #     'linear':linear.state_dict()
    # }, "./weights/model-{}.pth".format(epoch))




