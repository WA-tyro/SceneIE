import os

# hdr_path = 'C:/Users/WangAo/Desktop/rename/'
hdr_path = 'D:/Study/Dataset/Laval Indoor HDR Daset/'
hdr_names = [f for f in os.listdir(hdr_path) if f.endswith('.exr')]

txt = open('reset.txt','r')
names = txt.readlines()
for name in names:
    old_name=name.split(' ')[0]
    new_name_hdr = name.split(' ')[1].strip('\n')
    for name_hdr in hdr_names:
        if old_name == name_hdr.strip('.exr'):
            print(name_hdr)
            file_path = hdr_path + name_hdr
            newPhotoname = hdr_path + new_name_hdr + '.exr'
            os.rename(file_path, newPhotoname)