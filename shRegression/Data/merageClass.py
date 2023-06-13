import os
from shutil import copy,move

# ----------------------把maePre文件夹中的所有图片复制到Input文件夹中---------------------
folder_path = 'D:/Study/Dataset/input_hdr_test/'
dest_path = 'D:/Study/Dataset/test/'

folder_names = os.listdir(folder_path)
for folder_name in folder_names:
    file_path = folder_path+folder_name
    file_names = os.listdir(file_path)
    for file_name in file_names:
        copy(file_path+'/'+file_name,dest_path+file_name)
        print(file_name)



