# -*- coding: utf-8 -*-

# task 20170222 task d3 1

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import Conv1DLayer, InputLayer, BatchNormLayer, MaxPool1DLayer, DenseLayer, dropout

opt = {
	'batch_size': 1,
	'model_name': 'checkpoints/99.npz',
    'n_shuffle': 10,
    'n_categories': 2, 
    'sig_start': 5,
    'sig_end': 8,
    'samp_start': 0, 
    'samp_end': 875, 
    'input_shape': (None, 3, 875),
    'output_num_units': 2,
    'X_test': 'X_test1.npz',
    'y_test': 'y_test1.npz'
}

#network = InputLayer(shape=(None, 8, 1000))
#network = Conv1DLayer(network, num_filters=28, filter_size=5,
#	nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
#network = MaxPool1DLayer(network, pool_size=2)
##network = BatchNormLayer(network)
#network = Conv1DLayer(network, num_filters=56, filter_size=5,
#	nonlinearity=lasagne.nonlinearities.rectify)
#network = MaxPool1DLayer(network, pool_size=2)
##network = BatchNormLayer(network)
#network = Conv1DLayer(network, num_filters=112, filter_size=5, 
#                      nonlinearity=lasagne.nonlinearities.rectify)
#network = MaxPool1DLayer(network, pool_size=2)
##network = BatchNormLayer(network)
#network = DenseLayer(dropout(network, p=0.5), num_units=512,
#	nonlinearity=lasagne.nonlinearities.rectify)
#network = DenseLayer(dropout(network, p=0.5), num_units=256,
#	nonlinearity=lasagne.nonlinearities.rectify)
#network = DenseLayer(dropout(network, p=0.5), num_units=2, 
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

with np.load(opt['model_name']) as f:
	param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)

input_var = T.tensor3('input_var')
target_var = T.ivector('target_var')
prediction = lasagne.layers.get_output(network, input_var)
test_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)

test_fn = theano.function(inputs=[input_var, target_var], outputs=[prediction, test_acc])

signal = np.load(opt['X_test'])['arr_0']
labels = np.load(opt['y_test'])['arr_0']

print 'signal.shape', signal.shape
print 'labels.shape', labels.shape

ext_signal = np.ndarray((signal.shape[0]*opt['n_shuffle'], signal.shape[1], signal.shape[2]))
ext_labels = np.ndarray((labels.shape[0]*opt['n_shuffle']))

for i in xrange(opt['n_shuffle']):
    indices = np.arange(signal.shape[0])
    np.random.shuffle(indices)
    ext_signal[i*signal.shape[0]:(i+1)*signal.shape[0]] = signal[indices]
    ext_labels[i*labels.shape[0]:(i+1)*labels.shape[0]] = labels[indices]
    
print 'ext_signal.shape', ext_signal.shape
print 'ext_labels.shape', ext_labels.shape

prediction = np.ndarray((ext_signal.shape[0], opt['n_categories']), dtype=np.float32)
test_acc = []
for i in xrange(ext_signal.shape[0]):
    pred, acc = test_fn(np.asarray([ext_signal[i, opt['sig_start']:opt['sig_end'], opt['samp_start']:opt['samp_end']]], dtype=np.float32), np.asarray([ext_labels[i]], dtype=np.uint8))
    prediction[i] = pred
    test_acc.append(acc)

prediction = np.asarray(prediction, dtype=np.float32)
test_acc = np.asarray(test_acc, dtype=np.float32)
np.savez_compressed('result.npz', ext_signal=ext_signal, prediction=prediction, ext_labels=ext_labels, test_acc=test_acc)
print prediction.shape
print 'prediction_array:', prediction
print 'prediction:', np.argmax(prediction, axis=1)
print 'ext_labels:', ext_labels
print 'test_acc:', test_acc.mean()
