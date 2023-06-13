import os
from shutil import copy,move

# # ------------------------删除wraped中有的但laval中没得-------------------------------------
# file_path = 'D:/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/warpedHDROutputs/'
# delete_path = 'D:/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/warpedHDROutputs/delete/'
# file_names = [f for f in os.listdir(file_path) if f.endswith('.exr')]
# remove_file = open('removeList.txt','r')
# remove_names = [f.strip('\n') for f in remove_file.readlines()]
# for name in file_names:
#     if name.split('-')[0] in remove_names:
#         move(file_path+name,delete_path+name)

# # ------------------------删除laval中有的，但wraped中没得----------------------------------------
# file_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/'
# delete_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/delete/'
# file_names = [f for f in os.listdir(file_path) if f.endswith('.exr')]
# remove_file = open('removeList.txt','r')
# remove_names = [f.strip('\n') for f in remove_file.readlines()]
# for name in file_names:
#     if name.split('-')[0] in remove_names:
#         print(file_path+name)
#         move(file_path+name,delete_path+name)

# # -------------------删除laval中重复的，但是wraped中没有找到对应图片的exr文件--------------------------
# file_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/'
# delete_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/delete/'
# file_names = [f.strip('.exr') for f in os.listdir(file_path) if f.endswith('.exr')]
# remove_file = open('removeList.txt','r')
# remove_names = [f.strip('\n') for f in remove_file.readlines()]
# for name in file_names:
#     if name in remove_names:
#         move(file_path+name+'.exr',delete_path+name+'.exr')


# # -------------------------删除裁剪的hdrinput中错误的文件--------------------------------------
# # 首先删除那些命名重复的，防止裁剪过程中覆盖
# # 再删除laval中有但wraped中没有的
#
# inputdir_path = 'D:/Study/Dataset/input_hdr/'
# delete_path = 'D:/Study/Dataset/input_hdr/delete/'
# folder_names = [f for f in os.listdir(inputdir_path)]
# remove_file = open('removeList.txt','r')
# remove_names = [f.strip('\n') for f in remove_file.readlines()]
# n = 0
# for name in folder_names:
#     if name.split('-')[0] in remove_names:
#         print(name)
#         move(inputdir_path+name,delete_path+name)
#         n+=1
#         print(n)


# ---------------------删除hdrInput中有的，但lavaltrain中没的-----------------------------------
inputdir_path = 'D:/Study/Dataset/input_hdr/'
delete_path = 'D:/Study/Dataset/input_hdr/delete/'
laval_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/train/panoramaHDR/'
folder_names = [f for f in os.listdir(inputdir_path)]
laval_names = [f.split('.')[0] for f in os.listdir(laval_path)]
n = 0
for name in folder_names:
    if name not in laval_names:
        print(name)
        move(inputdir_path+name,delete_path+name)
        n+=1
        print(n)

