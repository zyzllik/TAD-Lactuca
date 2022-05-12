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
from pathlib import Path

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


def mlp_result(feature_y, feature_n, result_folder, hist_list = False, exclude=False, date=None):
    # (x_train_mlp, y_train_mlp), (x_test_mlp, y_test_mlp) = load_data.mlp(feature=feature_data, exclude=excluded_data)
    (x_train_mlp, y_train_mlp), (x_test_mlp, y_test_mlp) = load_data.csv_load(feature_y, feature_n, hist_mod_list=hist_list, exclusion=exclude)
    clf_mlp = KerasClassifier(build_fn=mlp, input=x_train_mlp, epochs=epochs, batch_size=batch_size, verbose=0)
    clf_mlp.fit(x_train_mlp, y_train_mlp, validation_data=(x_test_mlp, y_test_mlp)) # history = 
    y_pred_mlp = clf_mlp.predict_proba(x_test_mlp)[:, 1]
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test_mlp, y_pred_mlp)
    if hist_list is False:
        file_name = '{0}_MLP_full_model.npz'.format(date)
    else:
        file_name = '{0}_MLP_exclude_{1}'.format(date, '_'.join(hist_list))
    # np.savetxt("0502_results_MLP/{}".format(file_name), [fpr_mlp, tpr_mlp, y_pred_mlp, y_test_mlp], fmt='%.8f')
    if not result_folder.exists():
        result_folder.mkdir()
    np.savez(result_folder/"{}".format(file_name), fpr = fpr_mlp, tpr = tpr_mlp, pred = y_pred_mlp, target = y_test_mlp)
    print('*******' * 3, '\n\t AUC {} = '.format(hist_list), auc(fpr_mlp, tpr_mlp), '\n', '*******' * 3)



if __name__ == '__main__':
    print("Hello")
    mlp_result(feature_data="../../data/feature/feature_bin_10_400kb_201707131020.xlsx")
