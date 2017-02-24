# -*- coding: utf-8 -*-

import numpy as np

train = np.load('train.npz')['arr_0']
label = np.load('label.npz')['arr_0']

mask = np.asarray([True if l == 0 or l == 2 else False for l in label])
print mask
print label

label2 = label[mask]
label3 = np.asarray([1 if l == 2 else 0 for l in label2], dtype=np.uint8)
print label2
print label3
train2 = train[mask]

np.savez_compressed('label2.npz', label3)
np.savez_compressed('train2.npz', train2)
