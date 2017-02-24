# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import StratifiedKFold

train = np.load('train2.npz')['arr_0']
labels = np.load('label2.npz')['arr_0']
train_fake = np.arange(train.shape[0]*train.shape[1]).reshape((train.shape[0], train.shape[1]))

skf = StratifiedKFold(n_splits=3)

n_k_fold = 0

for train_index, test_index in skf.split(train_fake, labels):
    np.savez_compressed('X_train'+str(n_k_fold)+'.npz', np.asarray(train[train_index], dtype=np.float32))
    np.savez_compressed('y_train'+str(n_k_fold)+'.npz', np.asarray(labels[train_index], dtype=np.uint8))
    np.savez_compressed('X_test'+str(n_k_fold)+'.npz', np.asarray(train[test_index], dtype=np.float32))
    np.savez_compressed('y_test'+str(n_k_fold)+'.npz', np.asarray(labels[test_index], dtype=np.uint8))
    n_k_fold += 1
    