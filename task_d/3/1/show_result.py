# -*- coding: utf-8 -*-

import numpy as np

result = np.load('result.npz')

print 'test_acc:', result['test_acc'].mean()
print 'prediction:', np.argmax(result['prediction'], axis=1)
print 'ext_labels:', result['ext_labels'].astype(np.uint8)