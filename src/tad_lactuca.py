# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 9/7/17 10:39 AM
"""
import sys
import itertools
from pathlib import Path

workspace = './'
sys.path.append(workspace)


from model import cnn_model, mlp_model, rf_model, svm_model
# from utils import get_signal_plot
from utils.plots import *

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # print("If you want to use your data, please run script as:  \n\t  python3 tad_lcatuca.py ['the path to the data']")
        # feature_data = "cache/E017/feature/bin_10_400kb.xlsx"
        feature_pos= Path("/net/data.isilon/ag-cherrmann/echernova/model_input/K562/positives_K562.csv")
        feature_neg= Path("/net/data.isilon/ag-cherrmann/echernova/model_input/K562/negatives_K562.csv")
    else:
        feature_data = sys.argv[1]

    # not_available = ['E017-H3K4me3', 'E017-H3K4me2', 'E017-CTCF', 'E017-H3K9ac']
    # not_available = ['E017-H3K4me3', 'E017-H3K4me2', 'E017-H3K36me3',
    #    'E017-CTCF', 'E017-H3K27me3', 'E017-H3K9me3',
    #    'E017-H3K27ac', 'E017-H3K9ac', 'E017-H3K4me1']
    exclude_list = [False, ['H2AZF', 'H3K4me2', 'H3K4me3', 'H3K79me2', 'H3K9ac', 'H3K9me1', 'H3K9me3', 'H4K20me1']]
    # for i in range(1, 5):
    #     exclude_list += [j for j in itertools.combinations(not_available, i)]
    # print(exclude_list)
    # exclude_list = [False, ['E017-H3K9ac'], ['E017-H3K27ac'], ['E017-H3K9ac', 'E017-H3K27ac']]

    print("MLP...")
    result_folder_mlp = Path('/net/data.isilon/ag-cherrmann/echernova/model_output/K562')
    for exluded_feature in exclude_list:
        print(exluded_feature)
        mlp_model.mlp_result(feature_pos, feature_neg, result_folder_mlp, excluded_data=exluded_feature)
    plot_roc_folder(result_folder_mlp, result_folder_mlp/'0511_mlp_ROC_curve_K562_only_available_mods.png', 'ROC comparison: MLP on K562')
    # print("RF...")
    # result_folder_rf = Path('0502_results_all_combis_MLP_v2')
    # for exluded_feature in exclude_list:
    #     print(exluded_feature)
    #     rf_model.rf_result(feature_data, result_folder_rf, excluded_data=exluded_feature)

    


    # print("Plot...")
    # get_signal_plot.plot_dis(feature_file=feature_data, label_file=get_signal_plot.label_file, cell_line='IMR90',
    #                          feature_dimensional=get_signal_plot.feature_dimensional)
