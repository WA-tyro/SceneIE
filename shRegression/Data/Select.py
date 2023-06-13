import os

# -------------------讲图片和存储路径写入对应的txt文件-----------------------------

test_set = list(range(1, 32))
train_set = [_ for _ in range(1, 401) if _ not in test_set]


def getSHMean(shfile):
    with open(shfile) as f:
        first_line = f.readline()
    x = sum(map(float, first_line.split()))
    return x / 3


with open('list/train.txt', 'w') as f:
    for i in train_set:
        for j in range(512):
            rng = getSHMean('/home/wangao/PycharmProjects/code/data/sh/%04d/%03d.txt' % (i, j))
            # filter out images under/over exposure
            if 0.3 < rng < 4:
                f.write('/home/wangao/PycharmProjects/code/data/im/%04d/%03d_1.jpg ' % (i, j))
                f.write('/home/wangao/PycharmProjects/code/data/sh/%04d/%03d.txt\n' % (i, j))
                f.write('/home/wangao/PycharmProjects/code/data/im/%04d/%03d_0.jpg ' % (i, j))
                f.write('/home/wangao/PycharmProjects/code/data/sh/%04d/%03d.txt\n' % (i, j))
print('train.txt: done')
with open('list/test.txt', 'w') as f:
    for i in test_set:
        for j in range(512):
            rng = getSHMean('/home/wangao/PycharmProjects/code/data/sh/%04d/%03d.txt' % (i, j))
            # filter out images under/over exposure
            if 0.3 < rng < 4:
                f.write('/home/wangao/PycharmProjects/code/data/im/%04d/%03d_1.jpg ' % (i, j))
                f.write('/home/wangao/PycharmProjects/code/data/sh/%04d/%03d.txt\n' % (i, j))
                f.write('/home/wangao/PycharmProjects/code/data/im/%04d/%03d_0.jpg ' % (i, j))
                f.write('/home/wangao/PycharmProjects/code/data/sh/%04d/%03d.txt\n' % (i, j))
print('test.txt: done')


#
# with open('train.txt','r') as f:
#     lines = f.readlines()
#
# with open('./list/train.txt','w') as f:
#     n = 1
#     for line in lines:
#         if (n % 2 == 0):
#             f.write(line)
#         n += 1

