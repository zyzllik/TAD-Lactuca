# -*- coding:utf-8 -*-
import sys
import itertools
from pathlib import Path

workspace = './'
sys.path.append(workspace)


from model import mlp_model, rf_model, svm_model
# from utils import get_signal_plot
from utils.plots import *

if __name__ == '__main__':
    date = '0517'
    if len(sys.argv) == 1:
        feature_data = "cache/E017/feature/bin_10_400kb.xlsx"
    else:
        feature_data = sys.argv[1]

# Generating all combinations of following features to be excluded
not_available = ['H3K4me3', 'H3K4me2', 'CTCF', 'H3K9ac']
data_list = [False]
exclude = True
for i in range(1, 5):
    data_list += [j for j in itertools.combinations(not_available, i)]
print(data_list)

# MLP model
print("MLP...")
result_folder_mlp = Path('results/{0}_corrected_feature_exclusion'.format(date))
for exluded_features in data_list:
    print(exluded_features)
    mlp_model.mlp_result(feature_data, result_folder_mlp, hist_list=exluded_features, exclude=exclude, date=date, input_type='xlsx')
plot_roc_folder(result_folder_mlp, result_folder_mlp/'{0}_mlp_ROC_curve_excluded_features.png'.format(date), 'ROC comparison: MLP on original data')

# RF model
print("RF...")
result_folder_rf = Path('results/{0}_corrected_feature_exclusion'.format(date))
for exluded_features in data_list:
    print(exluded_features)
    rf_model.rf_result(feature_data, result_folder_rf, hist_list=exluded_features, exclude=exclude, date=date, input_type='xlsx')
plot_roc_folder(result_folder_rf, result_folder_rf/'{0}_rf_ROC_curve_excluded_features.png'.format(date), 'ROC comparison: RF on original data')
    