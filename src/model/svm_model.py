# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 10/7/17 1:23 PM
"""
from __future__ import print_function
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC

from src.utils import load_data


def svm_result(feature_data):
    (x_train_svm, y_train_svm), (x_test_svm, y_test_svm) = load_data.mlp(feature=feature_data)
    clf_svm = SVC(probability=True)
    clf_svm.fit(x_train_svm, y_train_svm)
    y_pred_svm = clf_svm.predict_proba(x_test_svm)[:, 1]
    fpr_svm, tpr_svm, _ = roc_curve(y_test_svm, y_pred_svm)
    print('*******' * 3, '\n\t AUC = ', auc(fpr_svm, tpr_svm), '\n', '*******' * 3)


if __name__ == '__main__':
    print("Hello")
    svm_result(feature_data="../../data/feature/feature_bin_10_400kb_201707131020.xlsx")
