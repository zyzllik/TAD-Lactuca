import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn import metrics
from matplotlib.lines import Line2D

def plot_roc(ax, file_path, data_name):
    file = np.load(file_path, 'r')
    data_auc = metrics.auc(file['fpr'],file['tpr'])
    data_name = data_name + ' AUC = {0:.4f}'.format(data_auc)
    ax.plot(file['fpr'], file['tpr'], label = data_name)

def plot_roc_folder(folder, output_name, title):
    file_names = os.listdir(folder)
    fig, ax = plt.subplots()

    # ax.set_prop_cycle(color=iter(plt.cm.rainbow(np.linspace(0, 1, len(file_names)))))

    ax.set_prop_cycle(color=[
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5'])

    for file in file_names:
        if file[:-4:-1] == 'zpn':
            path = folder / file
            if 'False' in file or 'full' in file:
                name = 'full model;'
            else:
                name = ', '.join(file.split('.')[0].split('_')[3:]) + ' excluded;'  
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
    folder_path = Path('0503_results_MLP_all') #RF
    # plot_roc_folder(folder_path, '0502_results_MLP_CTCF_v2/0502_roc_MLP.png', 'ROC comparison: MLP')
    file_names = os.listdir(folder_path)
    fig, ax = plt.subplots()

    for file_n in file_names:
        linewidth = 1.5
        marker = ','
        
        if file_n[:-4:-1] == 'zpn':
            path = folder_path / file_n
            if 'False' in file_n or 'full' in file_n:
                name = 'full model;'
                # color = 'black'
                # linewidth = 5
            else:
                name = ', '.join(file_n.split('.')[0].split('_')[3:]) + ' excluded;'
            # if 'CTCF' in file_n:
            #     if 'H3K9ac' in file_n:
            #         color = 'cyan'
            #     else:
            #         color = 'red'
            # else:
            #     if 'H3K9ac' in file_n:
            #         color = 'blue'
            #     else:
            #         color = 'gray'
          
            file = np.load(path, 'r')
            data_auc = metrics.auc(file['fpr'],file['tpr'])
            data_name = name + ' AUC = {0:.4f}'.format(data_auc)

            # ax.plot(file['fpr'], file['tpr'], label = data_name, color=color, linewidth=linewidth, linestyle=':')
            ax.plot(file['fpr'], file['tpr'], label = data_name, linewidth=linewidth, linestyle=':')

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('False negative rate')
    ax.set_title('ROC comparison: MLP')
    # custom_lines = [Line2D([0], [0], color='black', lw=4),
    #                 Line2D([0], [0], color='cyan', lw=4),
    #                 Line2D([0], [0], color='red', lw=4),
    #                 Line2D([0], [0], color='blue', lw=4),
    #                 Line2D([0], [0], color='gray', lw=4)
    #                 ]
    # ax.legend(custom_lines, ['Full model', '-CTCF -H3K9ac', '-CTCF', '-H3K9ac', '-other'])
    ax.legend()
    fig.set_size_inches(30, 30)
    fig.savefig(folder_path/'roc_MLP_all_combis_2.png')


