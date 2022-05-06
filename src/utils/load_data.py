# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 9/7/17 9:50 AM
"""

from keras import backend as K
# from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from PIL import Image
import pandas as pd
import numpy as np

from src.utils import get_signal_plot

bin_num = get_signal_plot.bin_number
histone_type = get_signal_plot.histone_type
num_classes = get_signal_plot.num_classes
img_rows, img_cols = histone_type, bin_num * 2 + 1
dpi = get_signal_plot.dpi
pic_path = get_signal_plot.pic_path
tad_num = get_signal_plot.tad_num
tad_not_num = get_signal_plot.tad_not_num


def matrix_to_image(data, num, flag):
    min_max_scaler = MinMaxScaler(feature_range=(0, 255))
    data = min_max_scaler.fit_transform(data)
    pic = Image.fromarray(data.astype(np.uint8))
    if flag:
        pic.save('%s/image/tad/%s_%s_bin.eps' % (pic_path, num, bin_num), format='eps', dpi=dpi)
    else:
        pic.save('%s/image/no_tad/%s_%s_bin.eps' % (pic_path, (int(num) - tad_num), bin_num), format='eps',
                 dpi=1000)


# def cnn(feature):
#     df_y = pd.read_excel(feature, sheetname='y').fillna(0)
#     df_n = pd.read_excel(feature, sheetname='n').fillna(0)
#     df = df_y.append(df_n)
#     row = [i for i in range(df.shape[0])]
#     index = [i for i in range(4, df.shape[1])]
#     data = np.empty((df.shape[0], 1, img_rows, img_cols), dtype="float32")
#     for current_row in row:
#         tmp_data = np.asarray(df.iloc[current_row, index], dtype=float)
#         current_data = np.reshape(tmp_data, [img_rows, img_cols])
#         if current_row >= tad_num:
#             flag = 0
#         else:
#             flag = 1
#         matrix_to_image(current_data, current_row, flag=flag)
#         data[current_row, :] = current_data
#     target = df.iloc[:, 3]
#     if K.image_data_format() == 'channels_first':
#         data = data.reshape(data.shape[0], 1, img_rows, img_cols)
#         input_shape = (1, img_rows, img_cols)
#     else:
#         data = data.reshape(data.shape[0], img_rows, img_cols, 1)
#         input_shape = (img_rows, img_cols, 1)
#     data = data.astype('float32')
#     target = to_categorical(target, num_classes)
#     x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7, random_state=49)
#     return (x_train, y_train), (x_test, y_test), input_shape



def mlp(feature, exclude = False):

    df_y = pd.read_excel(feature, sheet_name='y').fillna(0)
    df_n = pd.read_excel(feature, sheet_name='n').fillna(0)
    df = df_y.append(df_n)

    if exclude is not False:
        for excluded_input in exclude:
            filter_columns = [col_name for col_name in list(df.columns) if col_name.startswith(excluded_input)]
        df = df.drop(filter_columns, axis=1)
        print('columns: {}'.format(df.columns))
    
    index = [i for i in range(4, df.shape[1])]
    data = np.matrix(df.iloc[:, index])
    target = np.array(df.iloc[:, 3])
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7, random_state=49)
    return (x_train, y_train), (x_test, y_test)


def data(feature):
    df_y = pd.read_excel(feature, sheet_name='y').fillna(0)
    df_n = pd.read_excel(feature, sheet_name='n').fillna(0)
    df = df_y.append(df_n)
    index = [i for i in range(4, df.shape[1])]
    data = np.matrix(df.iloc[:, index])
    target = np.array(df.iloc[:, 3])
    return data, target


def genome_data(feature):
    df_y = pd.read_excel(feature, sheet_name='y').fillna(0)
    df_n = pd.read_excel(feature, sheet_name='n').fillna(0)
    df = df_y.append(df_n)
    row = [i for i in range(df.shape[0])]
    index = [i for i in range(4, df.shape[1])]
    data = np.matrix(df.iloc[:, index])
    # data = np.empty((df.shape[0], 1, img_rows, img_cols), dtype="float32")
    # for current_row in row:
    #     tmp_data = np.asarray(df.iloc[current_row, index], dtype=float)
    #     current_data = np.reshape(tmp_data, [img_rows, img_cols])
    #     data[current_row, :] = current_data
    return data


def get_label(data):
    label = []
    index = np.genfromtxt(data, dtype=str)
    for i in index:
        if i.split('_')[0].split('-')[1] not in label:
            label.append(i.split('_')[0].split('-')[1])
    return label


if __name__ == '__main__':
    pass
