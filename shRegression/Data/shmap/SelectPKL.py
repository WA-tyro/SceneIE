import os
import random
from shutil import copy,move

# 设置文件夹路径和测试集大小
folder_path_gt = '/media/wangao/DataDisk/Study/Dataset/Laval Indoor HDR Daset/IndoorHDRDataset2018/test/'
folder_path_inputs = '/home/wangao/PycharmProjects/Illumination_Estimation/EMLight/warped_Laval_Indoor_HDR_Dataset/pkl/'
dest_path = folder_path_inputs + 'test/'
# 获取所有png格式的文件名
file_names_gt = [f.split('-')[0] for f in os.listdir(folder_path_gt)]
file_names_inputs = os.listdir(folder_path_inputs)

for file_name in file_names_inputs:
    tem = file_name.split('-')[0]
    file_path = os.path.join(folder_path_inputs, file_name)
    if tem in file_names_gt:
        # copy(file_path,dest_path+file_name)
        move(file_path,dest_path+file_name)
        print(file_name)
