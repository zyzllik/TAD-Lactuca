# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 9/7/17 9:45 AM
"""
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve, auc

from matplotlib import pyplot as plt
import numpy as np
import os

from src.utils import load_data

batch_size = 256
num_classes = 2
epochs = 60
input_shape = (load_data.img_rows, load_data.img_cols, 1)

workspace = '/home/workspace/Python/TAD'


def cnn_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=2, activation='tanh', input_shape=input_shape))
    model.add(Conv2D(64, 2, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 2, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 2, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 2, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 2, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn_result(feature_data, plot=0):
    (x_train_cnn, y_train_cnn), (x_test_cnn, y_test_cnn), _ = load_data.cnn(feature=feature_data)
    clf_cnn = KerasClassifier(build_fn=cnn_model, epochs=epochs, batch_size=batch_size, verbose=0)
    clf_cnn.fit(x_train_cnn, y_train_cnn, validation_data=(x_test_cnn, y_test_cnn))
    y_pred_cnn = clf_cnn.predict_proba(x_test_cnn)[:, 1]
    fpr_cnn, tpr_cnn, _ = roc_curve(np.array(y_test_cnn[:, 1], dtype='uint8'), y_pred_cnn, pos_label=1)
    np.savetxt("../E017_new_cnn.txt", [fpr_cnn, tpr_cnn], fmt='%.8f')
    print('*******' * 3, '\n\t AUC = ', auc(fpr_cnn, tpr_cnn), '\n', '*******' * 3)
    if plot:
        plt.plot(fpr_cnn, tpr_cnn, label='CNN AUC=%0.3f' % (auc(fpr_cnn, tpr_cnn)))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    print("Hello")
    cnn_result(feature_data="../../data/feature/feature_bin_10_400kb_201707131020.xlsx")
