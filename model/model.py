#!/usr/bin/env python
# coding=utf-8

import math
import time
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
# import itertools
# from sklearn import preprocessing
# from operator import itemgetter
# from sklearn.metrics import mean_squared_error
# from math import sqrt
from keras.models import Sequential

from util.data_utils import load_data, get_data, get_test_data


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


def build_model2(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    # model.add(LSTM(32,input_shape=(layers[1], layers[0]), return_sequences=False))
    # model.add(Dropout(0.1))
    model.add(Dense(16, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    return model


def main():
    window = 5
    df = get_data('../data/国电头/press.xls')
    df_test = get_test_data('../data/国电头/press.xls')
    X_train, y_train, X_test, y_test = load_data(df[:-1], window)  #
    X_TEST, Y_TEST , _, __ = load_data(df_test[:-1],window)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    model = build_model2([4, window, 1])

    model.fit(
        X_train,
        y_train,
        batch_size=50,
        nb_epoch=500,
        validation_split=0.001,
        verbose=2)
    diff = []
    ratio = []
    p = model.predict(X_TEST)
    for u in range(len(y_test)):
        pr = p[u][0]
        ratio.append((y_test[u] / pr) - 1)
        diff.append(abs(y_test[u] - pr))
        # print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))

    import matplotlib.pyplot as plt2

    plt2.plot(p, color='red', label='prediction')
    plt2.plot(Y_TEST, color='blue', label='y_test')
    plt2.legend(loc='upper right')
    plt2.axis([0,150,-0.25,1.25])
    # p.axis([0.0, 100, -0.25, 1.25])
    plt2.show()


if __name__ == '__main__':
    main()
