# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 9/7/17 10:39 AM
"""
import sys
from pathlib import Path

workspace = './'
sys.path.append(workspace)


from model import mlp_model, rf_model
from utils.plots import *

if __name__ == '__main__':

    # Load inputs
    if len(sys.argv) == 3:
        cell_line = sys.argv[1]
        date = sys.argv[2]
        feature_pos= Path("/net/data.isilon/ag-cherrmann/echernova/model_input/{0}/positives_{1}.csv".format(cell_line, cell_line))
        feature_neg= Path("/net/data.isilon/ag-cherrmann/echernova/model_input/{0}/negatives_{1}.csv".format(cell_line, cell_line))
    else:
        print("Incorrect number of inputs!")

    # Testing the full model and the model with features available for microglia
    exclude = False # the list is included
    data_list = [False, ['H3K9me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K27ac', 'CTCF']] # data which is available for microglia
    
    # MLP model
    # print("MLP...")
    # result_folder_mlp = Path('/net/data.isilon/ag-cherrmann/echernova/model_output/{0}'.format(cell_line))
    # for exluded_features in data_list:
    #     print(exluded_features)
    #     mlp_model.mlp_result((feature_pos, feature_neg), result_folder_mlp, hist_list=exluded_features, exclude=exclude, date=date, input_type='csv')
    # plot_roc_folder(result_folder_mlp, result_folder_mlp/'{0}_mlp_ROC_curve_{1}_only_available_mods.png'.format(date, cell_line), 'ROC comparison: MLP on {0}'.format(cell_line))
    
    # RF
    print("RF...")
    result_folder_rf = Path('/net/data.isilon/ag-cherrmann/echernova/model_output/{0}'.format(cell_line))
    for exluded_features in data_list:
        print(exluded_features)
        rf_model.rf_result((feature_pos, feature_neg), result_folder_rf, hist_list=exluded_features, exclude=exclude, date=date, input_type='csv')
    plot_roc_folder(result_folder_rf, result_folder_rf/'{0}_rf_ROC_curve_{1}_only_available_mods.png'.format(date, cell_line), 'ROC comparison: RF on {0}'.format(cell_line))
