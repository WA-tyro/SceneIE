import os
import random

# ---------------------随机挑选200张当作test集，但是写的有问题--------------------
train_image = '/media/wangao/DataDisk/Study/Dataset/Laval Indoor HDR Daset/train/HDRInput/'
train_sh = '/media/wangao/DataDisk/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/SHGT/train/'
test_image = train_image.replace('train','test')
test_sh = train_sh.replace('train','test')

file_names = [f for f in os.listdir(train_image) if f.endswith('.exr')]

test_file_names = [f for f in os.listdir(test_image) if f.endswith('.exr')]

# 将文件名写入train.txt和test.txt
with open('./list/train.txt', 'w') as train_file, open('./list/test.txt', 'w') as test_file:
    for file_name in file_names:
        file_path = os.path.join(train_image, file_name)
        sh_path = os.path.join(train_sh, file_name.replace('.exr','.txt'))
        train_file.write(f'{file_path}*')
        train_file.write(f'{sh_path}\n')
    for test_name in test_file_names:
        test_path_iamge = os.path.join(test_image, test_name)
        test_sh_path = os.path.join(test_sh, test_name.replace('.exr', '.txt'))
        test_file.write(f'{test_path_iamge}*')
        test_file.write(f'{test_sh_path}\n')
