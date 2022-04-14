import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn import metrics

def plot_roc(ax, file_path, data_name):
    fpr, tpr, _, _ = np.loadtxt(file_path)
    data_auc = metrics.auc(fpr, tpr)
    data_name = data_name + ' AUC = {0:.4f}'.format(data_auc)
    ax.plot(fpr, tpr, label = data_name)

def plot_roc_folder(folder, output_name, title):
    file_names = os.listdir(folder)
    fig, ax = plt.subplots()

    for file in file_names:
        if file[:-4:-1] == 'txt':
            path = folder / file
            if 'False' in file:
                name = 'full model;'
            else:
                name = '_'.join(file.split('-')[1:])[:-4] + ' excluded;'  
            plot_roc(ax, path, name)

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('False negative rate')
    ax.set_title(title)
    ax.legend(prop={'size': 6})
    fig.tight_layout()
    fig.savefig(output_name)

def plot_precision_recall(ax, file_path, data_name):
    file = np.load(file_path, 'r')
    
    precision, recall, _ = metrics.precision_recall_curve(file['target'], file['pred'])
    aupr = metrics.average_precision_score(file['target'], file['pred'])
    data_name = data_name + ' AUPR = {0:.4f}'.format(aupr)
    ax.plot(recall, precision, label = data_name)

def plot_precision_recall_folder(folder, output_name, title):
    file_names = os.listdir(folder)
    fig, ax = plt.subplots()

    for file in file_names:
        if file[-4:] == '.npz':
            
            path = folder / file
            if 'False' in file or 'full' in file:
                name = 'full model;'
            else:
                name = '_'.join(file.split('-')[1:])[:-4] + ' excluded;'  
            plot_precision_recall(ax, path, name)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(prop={'size': 6})
    fig.tight_layout()
    fig.savefig(output_name)


if __name__ == '__main__':
    folder_path = Path('results_exclude_features_RF/')
    plot_precision_recall_folder(folder_path, 'results_exclude_features_RF/precision_recall_reduced_rf.png', 'Precision recall curve comparison: RF')


