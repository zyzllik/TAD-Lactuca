# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 11/23/17 10:10 PM
"""

import os
import datetime
import random
import numpy as np
import pandas as pd
from collections import defaultdict

from openpyxl import Workbook

from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

from src.utils import load_data

dpi = 100
tad_num = 3836
tad_not_num = 2208
num_classes = 2
bin_size = 40000
bin_number = 10
histone_type = 9
feature_dimensional = bin_number * 2 + 1
cell_type = "E017_new"
root_path = "/home/loopgan/workspace/Python/TAD-Lactuca"
pic_path = "%s/pic/%s" % (root_path, cell_type)
cache_path = "%s/cache/%s" % (root_path, cell_type)
if not os.path.isdir(pic_path):
    os.system('mkdir -p %s' % pic_path)
if not os.path.isdir(cache_path):
    os.system('mkdir -p %s' % cache_path)
interval_dir = "%s/cache/%s/down_up_bin_num_%s_%skb/" % (
    root_path, cell_type, bin_number, int(int(bin_number * bin_size) / 1000))
feature_file = "%s/cache/%s/feature/bin_%s_%skb.xlsx" % (
    root_path, cell_type, bin_number, int((bin_size * bin_number) / 1000))
label_file = "%s/data/feature_index.txt" % root_path
cell_type_dir = '/home/loopgan/data/TAD/%s/' % cell_type


def get_interval_different_resolution(interval_dir, bin_size, bin_number, cell_type):
    '''
    :param interval_dir: 计算出的位点区间存放的目录，default = $root_path/$interval_dir
    :param bin_number: 位点上下游bin的个数，default = 10
    :param bin_size: bin的大小，default = 40kb
    :return: 2208个boundary,2208个non-boundary,文件命名为tad_b_$num.bed或tad_not_b_$num.bed
    '''
    TAD, NonTAD = 1, 0
    if TAD:
        with open("%s/cache/%s/tad_boundary_center.txt" % (root_path, cell_type), "w") as r:
            with open("%s/data/%s.txt" % (root_path, cell_type)) as f:
                for line in f.readlines():
                    if line.split()[1] == line.split()[2]:
                        r.write(line.split()[0] + "\t" + str(int(line.split()[1]) - int(bin_size / 2)) + "\n")
                    else:
                        tmp_center = str(int((int(line.split()[1]) + int(line.split()[2])) / 2) - int(bin_size / 2))
                        r.write(line.split()[0] + "\t" + tmp_center + "\n")
        with open("%s/cache/%s/tad_boundary_center.txt" % (root_path, cell_type)) as f:
            line_num = 1
            for line in f.readlines():
                with open("%s/tad_b_%s.bed" % (interval_dir, line_num), "w") as r:
                    for i in range(-bin_number, bin_number + 1):
                        r.write(line.split()[0] + "\t" + str(int(line.split()[1]) + i * bin_size) + "\t" + str(
                            int(line.split()[1]) + (i + 1) * bin_size) + "\n")
                line_num += 1
    if NonTAD:
        with open("%s/cache/%s/tad_not_boundary_center.txt" % (root_path, cell_type), "w") as r:
            with open("%s/data/input_IMR90_TADBoundries.txt" % root_path) as f:
                for line in f.readlines():
                    if line.split()[-1] == "NB":
                        r.write(line.split()[0] + "\t" + str(int(line.split()[1]) + 50000 - int(bin_size / 2)) + "\n")
        with open("%s/cache/%s/tad_not_boundary_center.txt" % (root_path, cell_type)) as f:
            line_num = 1
            for line in f.readlines():
                with open("%s/tad_not_b_%s.bed" % (interval_dir, line_num), "w") as r:
                    for i in range(-bin_number, bin_number + 1):
                        tmp_start = int(line.split()[1]) + i * bin_size if int(
                            int(line.split()[1]) + i * bin_size) > 0 else 0
                        tmp_end = int(line.split()[1]) + (i + 1) * bin_size if int(
                            int(line.split()[1]) + (i + 1) * bin_size) > 0 else 0
                        r.write(line.split()[0] + "\t" + str(tmp_start) + "\t" + str(tmp_end) + "\n")
                line_num += 1


def get_interval_signal():
    tad_y = defaultdict(list)
    tad_index = []
    with open("%s/cache/%s/tad_boundary_center.txt" % (root_path, cell_type)) as f1:
        for line in f1.readlines():
            tad_y[line.split()[0]].append(int(line.split()[1]))
    count = 0
    with open("%s/cache/%s/tad_signal.txt" % (root_path, cell_type), 'w') as r:
        with open("%s/cache/%s/%s.txt" % (root_path, cell_type, cell_type)) as f1:
            for line in f1.readlines():
                if not line.startswith("chr#"):
                    current_name = line.split()[0]
                    current_start = int(line.split()[1])
                    for k, v in tad_y.items():
                        for i in v:
                            if current_name == k and current_start in range(i, i + bin_size):
                                r.write(line)
                                tad_index.append(count)
                count += 1
    with open("%s/cache/%s/tad_y.index" % (root_path, cell_type), 'w') as f:
        for i in tad_index:
            f.write(str(i) + '\n')


def gen_random_index_not_b():
    with open("%s/cache/%s/%s.txt" % (root_path, cell_type, cell_type)) as f:
        lines = len(f.readlines())
    tad_y = []
    with open("%s/cache/%s/tad_y.index" % (root_path, cell_type)) as f:
        for i in f.readlines():
            tad_y.append(int(i))
    rest_list = list(set(tad_y) ^ set([i for i in range(1, lines)]))
    rand_index = random.sample(rest_list, len(tad_y))
    with open("%s/cache/%s/%s.txt" % (root_path, cell_type, cell_type)) as f:
        lines = f.readlines()
        with open("%s/cache/%s/tad_not_b_signal.txt" % (root_path, cell_type), 'w') as r:
            for i in rand_index:
                r.write(lines[i])
        with open("%s/cache/%s/tad_not_all_signal.txt" % (root_path, cell_type), 'w') as r:
            for i in rest_list:
                print(i)
                r.write(lines[i])


def plot_dis(feature_file, label_file, cell_line, feature_dimensional):
    '''
    :param dir_interval: 各信号所在目录
    :param cell_type: 组蛋白的类型
    :return: 在pic/box目录返回各种信号正负样本的余弦相似度，以箱线图形式。
    '''
    if os.path.isfile(feature_file):
        print("The file already exists, %s will be passed" % plot_dis.__name__.upper())
        # return
    x = [i for i in range(-bin_number, bin_number + 1)]
    data, target = load_data.data(feature=feature_file)
    print(data.shape)
    # exit()
    label = load_data.get_label(data=label_file)
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    fig_1, axes_1 = plt.subplots(nrows=3, ncols=3, sharex=True)
    i_0 = 0
    legend_0 = 0
    for row in axes:
        for col in row:
            tad_b = np.mean(data[0:tad_num, i_0 * feature_dimensional:(i_0 + 1) * feature_dimensional], axis=0)
            not_tad_b = np.mean(data[tad_num:data.shape[0], i_0 * feature_dimensional:(i_0 + 1) * feature_dimensional],
                                axis=0)
            tad_b = MinMaxScaler().fit_transform(tad_b.reshape(-1, 1))
            not_tad_b = MinMaxScaler().fit_transform(not_tad_b.reshape(-1, 1))
            if not legend_0:
                y, = col.plot(x, tad_b, label='b')
                n, = col.plot(x, not_tad_b, label='not b')
                legend_0 = 1
            else:
                col.plot(x, tad_b, label='b')
                col.plot(x, not_tad_b, label='not b')
            col.set_title(label[i_0], size=8)
            col.set_ylim([-0.05, 1.05])
            col.plot([0, 0], [0, 1.05], linestyle='--')
            i_0 += 1
    i_1 = 0
    legend_1 = 0
    for row in axes_1:
        for col in row:
            tad_b = np.mean(data[0:tad_num, i_1 * feature_dimensional:(i_1 + 1) * feature_dimensional], axis=0).reshape(
                -1, 1) / 10000
            not_tad_b = np.mean(data[tad_num:data.shape[0], i_1 * feature_dimensional:(i_1 + 1) * feature_dimensional],
                                axis=0).reshape(-1, 1) / 10000
            if not legend_1:
                y_1, = col.plot(x, tad_b, label='b')
                n_1, = col.plot(x, not_tad_b, label='not b')
                legend_1 = 1
            else:
                col.plot(x, tad_b, label='b')
                col.plot(x, not_tad_b, label='not b')
            col.set_title(label[i_1], size=8)
            ymajorLocator = MultipleLocator(0.5)
            ymajorFormatter = FormatStrFormatter('%1.1f')
            yminorLocator = MultipleLocator(0.1)
            col.yaxis.set_major_locator(ymajorLocator)
            col.yaxis.set_major_formatter(ymajorFormatter)
            col.yaxis.set_minor_locator(yminorLocator)
            i_1 += 1
    fig.text(0.5, 0.03, 'relative distance from center', ha='center', size=10)
    fig.text(0.04, 0.5, 'normalized signal', va='center', rotation='vertical', size=10)
    fig.legend([y, n], ['Boundary', 'Non-Boundary'], prop={'size': 'small'})
    fig.savefig("%s/dis_all_feature/%s.eps" % (pic_path, bin_number), format='eps', dpi=dpi)

    # fig_1.tight_layout()
    fig_1.subplots_adjust(wspace=0.3)
    fig_1.text(0.5, 0.04, 'relative distance from center(bin)', ha='center', size=10)
    fig_1.text(0.04, 0.5, 'signal (scale 1:%s)' % (r'$10^4$'), va='center', rotation='vertical', size=10)
    fig_1.legend([y_1, n_1], ['Boundary', 'Non-Boundary'], prop={'size': 'small'})
    fig_1.savefig("%s/dis_all_feature/%s_not_nor.eps" % (pic_path, bin_number), format='eps', dpi=dpi)


if __name__ == '__main__':
    if not os.path.exists(interval_dir):
        os.mkdir(interval_dir)
    # get_interval_different_resolution(interval_dir=interval_dir, bin_size=bin_size, bin_number=bin_number,
    #                                   cell_type=cell_type)
    # get_interval_signal()
    gen_random_index_not_b()
    # plot_dis(feature_file=feature_file, label_file=label_file, cell_line=cell_type,
    #          feature_dimensional=feature_dimensional)
