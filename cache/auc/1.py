import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt
auc_dir = "./"
file_list = os.listdir(auc_dir)
file_list.sort()

for i in file_list:
    if i.endswith(".txt"):
        if "cnn" not in i:
            with open(auc_dir + i) as f:
                line = f.readlines()
                fpr = [float(i) for i in line[0].split()]
                tpr = [float(i) for i in line[1].split()]
                resolution = i.split(".")[0].upper()
                if i.startswith("bart"):
                    resolution = "IMR90_2012_HubPredictor"
                plt.plot(fpr, tpr, label='%s AUC=%0.3f' %
                         (resolution, auc(fpr, tpr)))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(prop={'size': 7}, loc=4)
pixel = plt.gcf()
pixel.savefig("./tmp.eps", format='eps', dpi=1000)
plt.close()
