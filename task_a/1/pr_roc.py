# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

#==============================================================================
# prediction & label
#==============================================================================

prediction = np.load('result.npz')['prediction']
labels = np.load('result.npz')['ext_labels']
classes = np.arange(np.unique(labels).shape[0])
colors = cycle(['aqua', 'navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

y_test = label_binarize(labels, classes=classes)
y_score = np.argmax(prediction, axis=1)

n_classes = 1

#==============================================================================
# p-r
#==============================================================================

precision = dict()
recall = dict()
average_precision = dict()

for i in xrange(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test, y_score)
    average_precision[i] = average_precision_score(y_test, y_score)

precision['micro'], recall['micro'], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
average_precision['micro'] = average_precision_score(y_test, y_score, 
                 average='micro')
    
#==============================================================================
# roc, auc
#==============================================================================
    
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in xrange(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

all_fpr = np.unique(np.concatenate([fpr[i] for i in xrange(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in xrange(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

#==============================================================================
# plot p-r
#==============================================================================
plt.figure()
plt.clf()
plt.plot(recall['micro'], precision['micro'], color='gold', lw=lw, 
         label='micro-average Precision-recall curve (area = {0:0.2f})'.format(average_precision['micro']))
for i, color in zip(xrange(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc='lower right')
plt.show()

#==============================================================================
# plot roc, auc
#==============================================================================
plt.figure()
plt.clf()
plt.plot(fpr['micro'], tpr['micro'], 
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc['micro']),
         color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr['macro'], tpr['macro'], 
         label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc['macro']),
         color='navy', linestyle=':', linewidth=4)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, 
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Flase Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.show()