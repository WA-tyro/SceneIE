import os
import random
from shutil import copy,move

# # ------------------将input文件夹中的照片重命名为对应的wraped名字，并存入EMLigt文件夹中------------------------
#
# # 这是复制文件版本
# wraped_path = 'D:/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/warpedHDROutputs/'
# input_path ='D:/Study/Dataset/Laval Indoor HDR Daset/train/HDRInput/'
# save_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/train/HDRInput/'
#
# wraped_names = [f.strip('.exr') for f in os.listdir(wraped_path) if f.endswith('.exr')]
# input_names = [f.strip('.png') for f in os.listdir(input_path) if f.endswith('.exr')]
# for name in wraped_names:
#     wraped_name = name.split('-')[0]
#     wraped_order = int(name.split('-')[2])
#     for input_name1 in input_names:
#         input_name = input_name1.split('-')[0]
#         input_order = int(input_name1.split('_')[1])
#         if wraped_name == input_name and ((input_order-1)*40 == wraped_order):
#             saved_path = save_path+name+'.exr'
#             os.rename(input_path+input_name1+'.exr', saved_path)




# -------------------------------这是直接改名版本--------------------------------
wraped_path = 'D:/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/warpedHDROutputs/test/'
input_path ='D:/Study/Dataset/test/'
wraped_names = [f for f in os.listdir(wraped_path) if f.endswith('.exr')]
input_names = [f for f in os.listdir(input_path) if f.endswith('.exr')]
for name in wraped_names:
    wraped_name = name.split('-')[0]
    wraped_order = int(name.split('-')[2])
    for input_name1 in input_names:
        input_name = input_name1.split('-')[0]
        input_order = int(input_name1.split('_')[1].strip('.exr'))
        if wraped_name == input_name and ((input_order-1)*40 == wraped_order):
            oldPath = input_path + input_name1
            newPath = input_path + name
            os.rename(oldPath,newPath)



