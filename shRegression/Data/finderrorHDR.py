import os
import random
from shutil import copy,move
# # -------------------找到laval重复命名的exr-------------------------
# hdr_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/'
# hdrNames = [f.split('-')[0] for f in os.listdir(hdr_path) if f.endswith('.exr')]
# length = len(hdrNames)
# for i in range(length):
#     for j in range(i+1,length):
#         if hdrNames[i] == hdrNames[j]:
#             print(hdrNames[i])


# # ----------------------找到wraped中有但laval中没得exr---------------------------
# hdr_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/'
# wraped_path = 'D:/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/warpedHDROutputs/'
#
# hdr_names = [f.split('-')[0] for f in os.listdir(hdr_path) if f.endswith('.exr')]
# wraped_names = [f.split('-')[0] for f in os.listdir(wraped_path) if f.endswith('.exr')]
# for wraped_name in wraped_names:
#     if wraped_name in hdr_names:
#         pass
#     else:
#         print(wraped_name)



# ----------------------找到laval中有但wraped中没得exr---------------------------
hdr_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/'
wraped_path = 'D:/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/warpedHDROutputs/'

hdr_names = [f.split('-')[0] for f in os.listdir(hdr_path) if f.endswith('.exr')]
wraped_names = [f.split('-')[0] for f in os.listdir(wraped_path) if f.endswith('.exr')]
for hdr_name in hdr_names:
    if hdr_name in wraped_names:
        pass
    else:
        print(hdr_name)