import os
import random
from shutil import copy,move

# -----------------在
folder_path = 'D:/Study/Dataset/LDRInput/warpedpre/'
dest_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/train/'
hdr_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/'

folder_names = [f for f in os.listdir(folder_path)]
file_names = [f for f in os.listdir(hdr_path) if f.endswith('.exr')]


for name in folder_names:
    for filename in file_names:
        if filename.split('-')[0] == name:
            move(hdr_path+filename,dest_path+filename)
