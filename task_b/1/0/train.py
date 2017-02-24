# -*- coding: utf-8 -*-

# 20160210 按group_labels进行训练
# 20170218 换数据集，anze，label=outcome， channel=8， sample=1000
# 20170221 5-folds x-validation
# 20170222 task 20170222 b1 0

import numpy as np

import theano
import theano.tensor as T

import lasagne

from lasagne.layers import Conv1DLayer, InputLayer, BatchNormLayer, MaxPool1DLayer, DenseLayer, dropout

opt = {
    'niter': 100,
    'n_shuffle': 50,
    'sig_start': 0,
    'sig_end': 2,
    'input_shape': (None, 2, 1000),
    'output_num_units': 2,
    'X_train': 'X_train0.npz',
    'y_train': 'y_train0.npz'
}

trains = np.load(opt['X_train'])['arr_0'][:, opt['sig_start']:opt['sig_end']]
labels = np.load(opt['y_train'])['arr_0']

print 'trains.shape:', trains.shape
print 'labels.shape:', labels.shape

ext_trains = np.ndarray((trains.shape[0]*opt['n_shuffle'], trains.shape[1], trains.shape[2]))
ext_labels = np.ndarray((labels.shape[0]*opt['n_shuffle']))

for i in xrange(opt['n_shuffle']):
    indices = np.arange(trains.shape[0])
    np.random.shuffle(indices)
    ext_trains[i*trains.shape[0]:(i+1)*trains.shape[0]] = trains[indices]
    ext_labels[i*labels.shape[0]:(i+1)*labels.shape[0]] = labels[indices]
    
print 'ext_trains.shape', ext_trains.shape
print 'ext_labels.shape', ext_labels.shape
#network = InputLayer(shape=(None, 8, 1000))
#network = Conv1DLayer(network, num_filters=28, filter_size=3,
#	nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
#network = MaxPool1DLayer(network, pool_size=2)
#network = Conv1DLayer(network, num_filters=56, filter_size=3,
#	nonlinearity=lasagne.nonlinearities.rectify)
#network = MaxPool1DLayer(network, pool_size=2)
#network = DenseLayer(dropout(network, p=0.5), num_units=256,
#	nonlinearity=lasagne.nonlinearities.rectify)
#network = DenseLayer(dropout(network, p=0.5), num_units=6, 
#	nonlinearity=lasagne.nonlinearities.softmax)

network = InputLayer(shape=opt['input_shape'])
network = Conv1DLayer(network, num_filters=28, filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
network = MaxPool1DLayer(network, pool_size=2)
network = BatchNormLayer(network)
network = Conv1DLayer(network, num_filters=56, filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify)
network = MaxPool1DLayer(network, pool_size=2)
network = BatchNormLayer(network)
network = Conv1DLayer(network, num_filters=112, filter_size=3, 
                      nonlinearity=lasagne.nonlinearities.rectify)
network = MaxPool1DLayer(network, pool_size=2)
network = BatchNormLayer(network)
network = DenseLayer(dropout(network, p=0.5), num_units=512,
	nonlinearity=lasagne.nonlinearities.rectify)
network = DenseLayer(dropout(network, p=0.5), num_units=256,
	nonlinearity=lasagne.nonlinearities.rectify)
network = DenseLayer(dropout(network, p=0.5), num_units=opt['output_num_units'], 
	nonlinearity=lasagne.nonlinearities.softmax)


input_var = T.tensor3('input_var')
target_var = T.ivector('target_var')

prediction = lasagne.layers.get_output(network, input_var)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adam(loss, params)

train_fn = theano.function(inputs=[input_var, target_var], outputs=loss, updates=updates)

#trains = np.asarray([trains[i] for i in xrange(trains.shape[0])]*10, 
#                     dtype=trains.dtype)
#labels = np.asarray([labels[i] for i in xrange(labels.shape[0])]*10,
#                     dtype=labels.dtype)

for epoch in xrange(opt['niter']):
    train_loss = 0.0
    train_num = 0.0
    for i in xrange(ext_trains.shape[0]):
        train_loss += train_fn(np.asarray([ext_trains[i]], dtype=np.float32), np.asarray([ext_labels[i]], dtype=np.uint8))
        train_num += 1
    if (epoch+1)%100 == 0:
        np.savez_compressed('checkpoints/'+str(epoch), 
                            *lasagne.layers.get_all_param_values(network))
    print 'train loss', train_loss/train_num