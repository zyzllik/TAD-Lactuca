# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 9/7/17 9:51 AM
"""
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve, auc
import numpy as np

from src.utils import load_data

## -1 when excluded
#input_shape = load_data.img_cols * (load_data.img_rows -1)
batch_size = 128
num_classes = 2
epochs = 60


def mlp(input):
    model = Sequential()
    model.add(Dense(512, input_shape=(input.shape[1],)))
    model.add(Activation('linear'))
    model.add(Dropout(0.6975))
    model.add(Dense(256))
    model.add(Activation('softplus'))
    model.add(Dropout(0.5153))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(Activation('linear'))
    model.add(Dropout(0.4252))
    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(Activation('hard_sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # model.summary()
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    # model.save('../cache/mlp.h5')
    # model.save_weights('../cache/mlp_weights.h5')
    return model


def mlp_result(feature_data, excluded_data = False):
    (x_train_mlp, y_train_mlp), (x_test_mlp, y_test_mlp) = load_data.mlp(feature=feature_data, exclude=excluded_data)
    clf_mlp = KerasClassifier(build_fn=mlp, input=x_train_mlp, epochs=epochs, batch_size=batch_size, verbose=0)
    clf_mlp.fit(x_train_mlp, y_train_mlp, validation_data=(x_test_mlp, y_test_mlp))
    y_pred_mlp = clf_mlp.predict_proba(x_test_mlp)[:, 1]
    print(y_pred_mlp[:10])
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test_mlp, y_pred_mlp)
    if excluded_data is False:
        file_name = 'MLP_full'
    else:
        file_name = 'MLP_exclude_{}.txt'.format('_'.join(excluded_data))
    np.savetxt("results_exclude_features_MLP/{}".format(file_name), [fpr_mlp, tpr_mlp], fmt='%.8f')
    print('*******' * 3, '\n\t AUC {} = '.format(excluded_data), auc(fpr_mlp, tpr_mlp), '\n', '*******' * 3)


if __name__ == '__main__':
    print("Hello")
    mlp_result(feature_data="../../data/feature/feature_bin_10_400kb_201707131020.xlsx")
