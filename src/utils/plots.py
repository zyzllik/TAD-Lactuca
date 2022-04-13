import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.metrics import auc

def plot_roc(axis, file_path, data_name):
    fpr, tpr = np.loadtxt(file_path)
    data_auc = auc(fpr, tpr)
    data_name = data_name + ' AUC = {0:.4f}'.format(data_auc)
    axis.plot(fpr, tpr, label = data_name)

def plot_roc_folder(folder, output_name, title):
    file_names = os.listdir(folder)
    fig, ax = plt.subplots()

    for file in file_names:
        if file[:-4:-1] == 'txt':
            path = folder / file
            if file[0] == 'M':
                name = file[17:-4] + ' excluded;'
            else:
                name = 'full model;'
            plot_roc(ax, path, name)

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('False negative rate')
    ax.set_title(title)
    ax.legend()
    fig.savefig(output_name)

if __name__ == '__main__':
    folder_path = Path('results_exclude_features/')
    plot_roc_folder(folder_path, 'roc_reduced_mlp.png', 'ROC comparison of full and reduced models MLP')


