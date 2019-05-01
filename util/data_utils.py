#！/usr/bin/env python
#-*- coding: utf-8 -*-
#@Author:Shiyu Xue
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler


ss = MinMaxScaler()


def get_data(data_name):
    df=pd.read_excel(data_name)
    values = df.values
    values = values[1:, 30:34]
    values = ss.fit_transform(values)
    return values

def get_test_data(data_name):
    df = pd.read_excel(data_name)
    values = df.values
    values = values[1:, 35:39]
    values = ss.fit_transform(values)
    return values


def load_data(data,seq_len):
    data = pd.DataFrame(data)
    amount_of_features = len(data.columns)
    data = data.as_matrix()  # pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    print("load_data之后的总数据形状是：", result.shape)
    row = round(0.8 * result.shape[0])
    train = result[:int(row), :]
    print("~~~~~~~~~~~~~~~~~~~~~~")
    print("train的形状是:",train.shape)
    print("~~~~~~~~~~~~~~~~~~~~~~~")
    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]