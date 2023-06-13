import os
import random
from shutil import copy
from tqdm import tqdm

# ------------------------把wraped的hdr图片存入对应的文件夹------------------------------
folder_path = 'D:/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/warpedHDROutputs/jpg/'
dest_path = 'D:/Study/Dataset/LDRInput/warpedpre/'


file_names = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]


for file_name in tqdm(file_names):
    file_path = os.path.join(folder_path, file_name)
    file_folder = dest_path + file_name.split('-')[0]
    if os.path.exists(file_folder) is False:
        os.makedirs(file_folder)
    save_path = file_folder + '/' + file_name
    copy(file_path, save_path)

