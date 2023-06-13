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

checkpoints_path = tb_writer.log_dir + "/weights"
if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)
continueTrain = utils.continueTrain
epochs = 100

train_data, total_train = utils.creat_dataloader('./Data/list/train.txt')
Eval_data, total_eval = utils.creat_dataloader('./Data/list/test.txt')

model = models_vit.__dict__['vit_base_patch16'](
        img_size= 256,
        num_classes=48,
        drop_path_rate=0.1,
        global_pool=False,
    )

model.to(device)
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

    torch.save(model.state_dict(), checkpoints_path + "/model-{}.pth".format(epoch))





