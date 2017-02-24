# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


#X_train = np.load('X_train0.npz')['arr_0']
#y_train = np.load('y_train0.npz')['arr_0']
#X_test = np.load('X_test0.npz')['arr_0']
#y_test = np.load('y_test0.npz')['arr_0']
#
#print 'y_train:', y_train
#print 'y_test:', y_test

#for i in xrange(6, 7):
#    for j in xrange(X_test.shape[1]):
#        plt.subplot(X_test.shape[1], 1, j+1)
#        plt.plot(X_test[i, j])
#        
#plt.show()

#for j in xrange(X_test.shape[1]):
#    plt.subplot(X_test.shape[1], 1, j+1)
#    plt.plot(X_test[0, j], lw=2, label='label:0, pred:4')
#    plt.plot(X_test[1, j], lw=2, label='label:0')
#    plt.plot(X_test[10, j], lw=2, label='label:4')
#    
#plt.legend(loc='lower left')
#plt.show()
#
#plt.plot(X_test[0, 0], lw=2, label='label:0, pred:4')
#plt.plot(X_test[1, 0], lw=2, label='label:0')
#plt.plot(X_test[10, 0], lw=2, label='label:4')
#    
#plt.legend(loc='lower left')
#plt.show()

#for i in xrange(X_train.shape[0]):
#    if y_train[i] != 5:
#        continue
#    plt.subplot(2, 1, 1)
#    plt.plot(X_train[i, 0], lw=2)
#    plt.subplot(2, 1, 2)
#    plt.plot(X_train[i, 1], lw=2)
#
#plt.show()

train = np.load('train.npz')['arr_0']
label = np.load('label.npz')['arr_0']

for i in xrange(train.shape[0]):
    if label[i] == 0:     
        plt.subplot(2, 1, 1)
        plt.plot(train[i, 0], lw=2, color='gold')
        plt.subplot(2, 1, 2)
        plt.plot(train[i, 1], lw=2, color='gold')
    elif label[i] == 3:
        plt.subplot(2, 1, 1)
        plt.plot(train[i, 0], lw=2, color='navy')
        plt.subplot(2, 1, 2)
        plt.plot(train[i, 1], lw=2, color='navy')
plt.show()

for i in xrange(train.shape[0]):
    if label[i] == 0:     
        plt.subplot(3, 1, 1)
        plt.plot(train[i, 2], lw=2, color='gold')
        plt.subplot(3, 1, 2)
        plt.plot(train[i, 3], lw=2, color='gold')
        plt.subplot(3, 1, 3)
        plt.plot(train[i, 4], lw=2, color='gold')
    elif label[i] == 3:
        plt.subplot(3, 1, 1)
        plt.plot(train[i, 2], lw=2, color='navy')
        plt.subplot(3, 1, 2)
        plt.plot(train[i, 3], lw=2, color='navy')
        plt.subplot(3, 1, 3)
        plt.plot(train[i, 4], lw=2, color='navy')
plt.show()

for i in xrange(train.shape[0]):
    if label[i] == 0:     
        plt.subplot(3, 1, 1)
        plt.plot(train[i, 5], lw=2, color='gold')
        plt.subplot(3, 1, 2)
        plt.plot(train[i, 6], lw=2, color='gold')
        plt.subplot(3, 1, 3)
        plt.plot(train[i, 7], lw=2, color='gold')
    elif label[i] == 3:
        plt.subplot(3, 1, 1)
        plt.plot(train[i, 5], lw=2, color='navy')
        plt.subplot(3, 1, 2)
        plt.plot(train[i, 6], lw=2, color='navy')
        plt.subplot(3, 1, 3)
        plt.plot(train[i, 7], lw=2, color='navy')
plt.show()

colors = ['gold', 'navy', 'aqua', 'darkorange', 'green', 'blue']
for i in xrange(train.shape[0]):
    plt.plot(train[i, 0], lw=2, color=colors[label[i].astype(np.uint8)])    
plt.legend(loc='lower left')
plt.show()
#
#for i in xrange(train.shape[0]):
#    if label[i] != 5:
#        continue
#    else:
#        for j in xrange(train.shape[1]):
#            plt.subplot(train.shape[1], 1, j+1)
#            plt.plot(train[i, j], lw=2)
#            
#plt.show()