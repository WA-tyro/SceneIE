import os
import random
from shutil import copy,move
#
# # -------------------根据laval中的test，将wraped中的也挑选出对应的test数据集-------------------
# folder_path_hdr = 'D:/Study/Dataset/Laval Indoor HDR Daset/test/'
# folder_path_wraped = 'D:/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/hdrInputs/'
# dest_path = folder_path_wraped + 'test/'
#
# file_names_hdr = [f.split('-')[0] for f in os.listdir(folder_path_hdr)]
# file_names_wraped = os.listdir(folder_path_wraped)
#
# for name_hdr in file_names_hdr:
#     for name_wraped in file_names_wraped:
#         if name_hdr == name_wraped.split('-')[0]:
#             move(folder_path_wraped+name_wraped,dest_path+name_wraped)
#             print(name_wraped)


# -------------------根据laval中的train，将wraped中的也挑选出对应的train数据集-------------------
folder_path_hdr = 'D:/Study/Dataset/Laval Indoor HDR Daset/train/panoramaHDR/'
folder_path_wraped = 'D:/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/warpedHDROutputs/'
dest_path = folder_path_wraped + 'train/'

file_names_hdr = [f.split('-')[0] for f in os.listdir(folder_path_hdr) if f.endswith('.exr')]
file_names_wraped = [f for f in os.listdir(folder_path_wraped) if f.endswith('.exr')]

for name_hdr in file_names_hdr:
    for name_wraped in file_names_wraped:
        if name_hdr == name_wraped.split('-')[0]:
            move(folder_path_wraped+name_wraped,dest_path+name_wraped)
            print(name_wraped)
