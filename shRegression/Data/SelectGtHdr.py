import os
import random
from shutil import copy,move

# ---------------------随机挑选200张hdr当作test集--------------------
folder_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/train/panoramaHDR/'
dest_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/test/'
test_size = 200

# 获取所有png格式的文件名
file_names = [f for f in os.listdir(folder_path) if f.endswith('.exr')]

# 随机选择测试集文件名
test_file_names = random.sample(file_names, test_size)


for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    if file_name in test_file_names:
        # copy(file_path,dest_path+file_name)
        move(file_path,dest_path+file_name)
        print(file_name)
