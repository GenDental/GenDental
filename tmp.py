import numpy as np
a = np.load('files/zj/train.npy')
print(len(a))
b = np.load('files/alignment5x/train1.npy')
c = np.load('/data3/leics/dataset/GenDental/merged_alignmentg_5x_2/608.npz')
d = np.load('/data3/leics/GenDental/files/alignment_zj/train1.npy')
print(len(b))
print(len(d))
for i in b:
    if i not in d:
        print(i)
for i in d:
    if i not in b:
        print(i)
