# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 9/7/17 10:39 AM
"""
import sys
import itertools

workspace = './'
sys.path.append(workspace)


from src.model import cnn_model, mlp_model, rf_model, svm_model
from src.utils import get_signal_plot

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("If you want to use your data, please run script as:  \n\t  python3 tad_lcatuca.py ['the path to the data']")
        feature_data = "cache/E017/feature/bin_10_400kb.xlsx"
        # feature_data = "../cache/E017_new/feature/all.xlsx"
    else:
        feature_data = sys.argv[1]

    not_available = ['E017-H3K4me3', 'E017-H3K4me2', 'E017-CTCF', 'E017-H3K9ac']
    exclude_list = [False]
    for i in range(1, 5):
        exclude_list += [j for j in itertools.combinations(not_available, i)]

    # print("MLP...")
    # for exluded_feature in exclude_list:
    #     print(exluded_feature)
    #     mlp_model.mlp_result(feature_data, excluded_data=exluded_feature)

    print("RF...")
    for exluded_feature in exclude_list:
        print(exluded_feature)
        rf_model.rf_result(feature_data, excluded_data=exluded_feature)


    # print("Plot...")
    # get_signal_plot.plot_dis(feature_file=feature_data, label_file=get_signal_plot.label_file, cell_line='IMR90',
    #                          feature_dimensional=get_signal_plot.feature_dimensional)
