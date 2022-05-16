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


from model import mlp_model, rf_model, svm_model
# from utils import get_signal_plot
from utils.plots import *

if __name__ == '__main__':
    # if len(sys.argv) == 1:
        # print("If you want to use your data, please run script as:  \n\t  python3 tad_lcatuca.py ['the path to the data']")
        # feature_data = "cache/E017/feature/bin_10_400kb.xlsx"
    
    if len(sys.argv) == 3:
        cell_line = sys.argv[1]
        date = sys.argv[2]
        feature_pos= Path("/net/data.isilon/ag-cherrmann/echernova/model_input/{0}/positives_{1}.csv".format(cell_line, cell_line))
        feature_neg= Path("/net/data.isilon/ag-cherrmann/echernova/model_input/{0}/negatives_{1}.csv".format(cell_line, cell_line))
    else:
        print("Incorrect number of inputs!")
        # feature_data = sys.argv[1]

    ## ---- Test different cell lines --- ##

    exclude = False # the list is included
    data_list = [False, ['H3K9me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K27ac', 'CTCF']] # data which is available for microglia
    print("MLP...")
    result_folder_mlp = Path('/net/data.isilon/ag-cherrmann/echernova/model_output/{0}'.format(cell_line))
    for exluded_features in data_list:
        print(exluded_features)
        mlp_model.mlp_result((feature_pos, feature_neg), result_folder_mlp, hist_list=exluded_features, exclude=exclude, date=date, input_type='csv')
    plot_roc_folder(result_folder_mlp, result_folder_mlp/'{0}_mlp_ROC_curve_{1}_only_available_mods.png'.format(date, cell_line), 'ROC comparison: MLP on {0}'.format(cell_line))
    
    ## ---------------------------------- ##

    ## ---- Test original data with excluded features ---- ##
    # not_available = ['H3K4me3', 'H3K4me2', 'CTCF', 'H3K9ac']
    # data_list = [False]
    # exclude = True
    # for i in range(1, 5):
    #     data_list += [j for j in itertools.combinations(not_available, i)]
    # print(data_list)

    # print("MLP...")
    # result_folder_mlp = Path('/net/data.isilon/ag-cherrmann/echernova/model_output/{0}'.format(cell_line))
    # for exluded_features in data_list:
    #     print(exluded_features)
    #     mlp_model.mlp_result(feature_pos, feature_neg, result_folder_mlp, hist_list=exluded_features, exclude=exclude, date=date)
    # plot_roc_folder(result_folder_mlp, result_folder_mlp/'{0}_mlp_ROC_curve_{1}_only_available_mods.png'.format(date, cell_line), 'ROC comparison: MLP on {0}'.format(cell_line))
    
    ## --------------------------------------------------- ##


    # print("RF...")
    # result_folder_rf = Path('0502_results_all_combis_MLP_v2')
    # for exluded_feature in exclude_list:
    #     print(exluded_feature)
    #     rf_model.rf_result(feature_data, result_folder_rf, excluded_data=exluded_feature)

    


    # print("Plot...")
    # get_signal_plot.plot_dis(feature_file=feature_data, label_file=get_signal_plot.label_file, cell_line='IMR90',
    #                          feature_dimensional=get_signal_plot.feature_dimensional)
