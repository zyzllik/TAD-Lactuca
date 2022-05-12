# -*- coding:utf-8 -*-
"""
@author: loopgan
@time: 9/7/17 9:50 AM
"""


#from keras import backend as K
# from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler

# from PIL import Image
import pandas as pd
import numpy as np

#from src.utils import get_signal_plot

# bin_num = get_signal_plot.bin_number
# histone_type = get_signal_plot.histone_type
# num_classes = get_signal_plot.num_classes
# img_rows, img_cols = histone_type, bin_num * 2 + 1
# dpi = get_signal_plot.dpi
# pic_path = get_signal_plot.pic_path
# tad_num = get_signal_plot.tad_num
# tad_not_num = get_signal_plot.tad_not_num


# def matrix_to_image(data, num, flag):
#     min_max_scaler = MinMaxScaler(feature_range=(0, 255))
#     data = min_max_scaler.fit_transform(data)
#     pic = Image.fromarray(data.astype(np.uint8))
#     if flag:
#         pic.save('%s/image/tad/%s_%s_bin.eps' % (pic_path, num, bin_num), format='eps', dpi=dpi)
#     else:
#         pic.save('%s/image/no_tad/%s_%s_bin.eps' % (pic_path, (int(num) - tad_num), bin_num), format='eps',
#                  dpi=1000)


def csv_load(feature_y, feature_n, hist_mod_list=False, exclusion=False):

    df_y = pd.read_csv(feature_y).fillna(0)
    df_n = pd.read_csv(feature_n).fillna(0)
    # first column: index from R, second column:tad_ids --> need to be removed
    df_y.drop(['Unnamed: 0', 'tad_id'], axis=1,  inplace = True)
    df_n.drop(['Unnamed: 0', 'tad_id'], axis=1,  inplace = True)
    # append label column
    df_y['label'] = 1
    df_n['label'] = 0
    df = pd.concat([df_y, df_n], axis=0)

    if hist_mod_list is not False:
        if exclusion:
            drop_columns = []
            for hist_mod in hist_mod_list:
                drop_columns += [col_name for col_name in list(df.columns) if hist_mod in col_name]
            df = df.drop(drop_columns, axis=1)
        elif not exclusion: # aka inclusion
            keep_columns = ['chr', 'start', 'end']
            for hist_mod in hist_mod_list:
                keep_columns += [col_name for col_name in list(df.columns) if hist_mod in col_name]
            keep_columns += ['label']
            df = df[keep_columns]
        
    print('columns: {}'.format(df.columns))
    
    data = np.matrix(df.iloc[:,3:-1])
    target = np.array(df['label'])
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7, random_state=49)
    return (x_train, y_train), (x_test, y_test)

def xlsx_load(feature, hist_mod_list=False, exclusion=False):

    df_y = pd.read_excel(feature, sheet_name='y').fillna(0)
    df_n = pd.read_excel(feature, sheet_name='n').fillna(0)
    df = df_y.append(df_n)

    if hist_mod_list is not False:
        if exclusion:
            drop_columns = []
            for hist_mod in hist_mod_list:
                drop_columns += [col_name for col_name in list(df.columns) if hist_mod in col_name]
            df = df.drop(drop_columns, axis=1)
        elif not exclusion: # aka inclusion
            keep_columns = ['chr', 'start', 'end']
            for hist_mod in hist_mod_list:
                keep_columns += [col_name for col_name in list(df.columns) if hist_mod in col_name]
            keep_columns += ['label']
            df = df[keep_columns]
        
    print('columns: {}'.format(df.columns))
    
    index = [i for i in range(4, df.shape[1])]
    data = np.matrix(df.iloc[:, index])
    target = np.array(df.iloc[:, 3])
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7, random_state=49)
    return (x_train, y_train), (x_test, y_test)


# def data(feature):
#     df_y = pd.read_excel(feature, sheet_name='y').fillna(0)
#     df_n = pd.read_excel(feature, sheet_name='n').fillna(0)
#     df = df_y.append(df_n)
#     index = [i for i in range(4, df.shape[1])]
#     data = np.matrix(df.iloc[:, index])
#     target = np.array(df.iloc[:, 3])
#     return data, target


# def genome_data(feature):
#     df_y = pd.read_excel(feature, sheet_name='y').fillna(0)
#     df_n = pd.read_excel(feature, sheet_name='n').fillna(0)
#     df = df_y.append(df_n)
#     row = [i for i in range(df.shape[0])]
#     index = [i for i in range(4, df.shape[1])]
#     data = np.matrix(df.iloc[:, index])
#     # data = np.empty((df.shape[0], 1, img_rows, img_cols), dtype="float32")
#     # for current_row in row:
#     #     tmp_data = np.asarray(df.iloc[current_row, index], dtype=float)
#     #     current_data = np.reshape(tmp_data, [img_rows, img_cols])
#     #     data[current_row, :] = current_data
#     return data


# def get_label(data):
#     label = []
#     index = np.genfromtxt(data, dtype=str)
#     for i in index:
#         if i.split('_')[0].split('-')[1] not in label:
#             label.append(i.split('_')[0].split('-')[1])
#     return label


if __name__ == '__main__':
    (x_tr, y_tr), (x_t, y_t) = csv_load("../../../K562/positives_K562.csv", "../../../K562/negatives_K562.csv")
