# -*- coding: utf-8 -*-

import numpy as np

train = np.loadtxt('CNNGolf1000Anze20170220.txt', delimiter=';')
label = np.loadtxt('CNNGolf1000Anze20170220_label.txt')

train2 = np.ndarray((85, 9, 1000), dtype=np.float32)

for i in xrange(85):
    for j in xrange(9):
        train2[i, j] = train[i*1000:(i+1)*1000, j]
        
np.savez_compressed('train.npz', train2)
np.savez_compressed('label.npz', label)