# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 9/7/17 3:23 PM
"""
from __future__ import print_function
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.utils import load_data
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

param_search = {'n_estimator': range(100, 1000, 100), 'max_depth': range(1, 5), 'min_samples_split': range(2, 10),
                'min_samples_leaf': range(40, 60, 2)}
rf = RandomForestClassifier()

clf = GridSearchCV(estimator=rf, param_grid=param_search, scoring='roc_auc', cv=5)


def rf_result(feature_data, excluded_data = False):
    (x_train_mlp, y_train_mlp), (x_test_mlp, y_test_mlp) = load_data.mlp(feature=feature_data, exclude=excluded_data)
    # param_search = {'n_estimators': range(100, 1000, 100), 'max_depth': range(1, 10), 'min_samples_split': range(2, 10),
    #                 'min_samples_leaf': range(1, 60, 2)}
    # param_search = {'n_estimators': range(100, 10000, 100)}
    # rf = RandomForestClassifier()
    # clf_rf = GridSearchCV(estimator=rf, param_grid=param_search, scoring='roc_auc', cv=5)
    # clf_rf.fit(x_train_mlp, y_train_mlp)
    # print(clf_rf.best_estimator_)
    #
    # all_data = load_data.genome_data(feature=feature_data.replace('bin_10_400kb', 'all'))
    # clf_rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=9,
    #                                 max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07,
    #                                 min_samples_split=7, min_weight_fraction_leaf=0.0,
    #                                 n_estimators=200, n_jobs=1, oob_score=False, random_state=None, verbose=0,
    #                                 warm_start=False)
    clf_rf = RandomForestClassifier(n_estimators=500)
    # print(clf_rf)
    clf_rf.fit(x_train_mlp, y_train_mlp)
    y_pred_rf = clf_rf.predict_proba(x_test_mlp)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test_mlp, y_pred_rf)
    if excluded_data is False:
        file_name = 'RF_full'
    else:
        file_name = 'RF_exclude_{}.txt'.format('_'.join(excluded_data))
    np.savetxt("results_exclude_features_RF/{}".format(file_name), [fpr_rf, tpr_rf], fmt='%.8f')
    print('*******' * 3, '\n\t AUC = ', auc(fpr_rf, tpr_rf), '\n', '*******' * 3)
    #
    # zz = clf_rf.predict(x_test_mlp)
    # print(classification_report(y_test_mlp, zz))
    # print('acc', accuracy_score(y_test_mlp, zz))
    # print(precision_recall_fscore_support(y_test_mlp, zz))
    # imp = clf_rf.predict_proba(all_data)[:, 1]
    # np.savetxt('./genome_wide_prob.txt', imp, fmt='%0.8f')


def svm_result(feature_data):
    (x_train_svm, y_train_svm), (x_test_svm, y_test_svm) = load_data.mlp(feature=feature_data)
    clf_svm = SVC(probability=True)
    clf_svm.fit(x_train_svm, y_train_svm)
    y_pred_svm = clf_svm.predict_proba(x_test_svm)[:, 1]
    fpr_svm, tpr_svm, _ = roc_curve(y_test_svm, y_pred_svm)
    print('*******' * 3, '\n\t AUC = ', auc(fpr_svm, tpr_svm), '\n', '*******' * 3)


if __name__ == '__main__':
    print("Hello")
    rf_result(feature_data="../../data/feature/feature_bin_10_400kb_201707131020.xlsx")
