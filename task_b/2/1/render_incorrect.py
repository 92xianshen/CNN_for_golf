# -*- coding: utf-8 -*-

import numpy as np

opt = {
        'n_shuffle': 10
        }

signal = np.load('X_test1.npz')['arr_0']
labels = np.load('y_test1.npz')['arr_0']
ext_signal = np.load('result.npz')['ext_signal']
ext_labels = np.load('result.npz')['ext_labels'].astype(np.uint8)
prediction = np.load('result.npz')['prediction']

pred_labels = np.argmax(prediction, axis=1)
mask = np.bitwise_not(np.equal(ext_labels, pred_labels))

print 'incorrect:', np.unique(ext_signal[mask][:, -1, 0])
print 'Fade swings in X_test0:', np.unique(signal[labels.astype(np.bool)][:, -1, 0])

ext_signal = ext_signal.reshape((opt['n_shuffle'], -1, ext_signal.shape[1], ext_signal.shape[2]))
ext_labels = ext_labels.reshape((opt['n_shuffle'], -1))
pred_labels = pred_labels.reshape((opt['n_shuffle'], -1))

for i in xrange(opt['n_shuffle']):
    mask = np.bitwise_not(np.equal(ext_labels[i], pred_labels[i]))
    print i, ' incorrect:', np.unique(ext_signal[i][mask][:, -1, 0])
