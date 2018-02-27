import os
import numpy as np
from numpy.random import permutation
from matplotlib import pyplot as plt
from PIL import Image
import itertools

def show_train_history(train_history, train, val):
    plt.plot(train_history.history[train],'r', label='train acc')
    plt.plot(train_history.history[val], 'b', label='val acc')
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     (This function is copied from the scikit docs.)
#     """
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')