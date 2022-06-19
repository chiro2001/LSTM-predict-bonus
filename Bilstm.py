# -*- coding: utf-8 -*-
"""
    Tensorflow-gpu 1.12.0
    cuda 9.0
    cudnn 7.6.5
    keras 2.1.6
    h5py 2.9.0
"""
from __future__ import print_function
import cmath
import math

from keras.layers import (
    Input,
    Activation,
)
from keras.regularizers import l2
import keras
from keras.models import Model, Sequential, Input
from keras.layers import LSTM, Dropout, Dense, Activation, Bidirectional
import os
import pickle
import sys
import time
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
np.random.seed(1337)  # for reproducibility
# parameters
nb_epoch = 100  # number of epoch at training stage. To find a nice epochs in the valid dataset.
batch_size = 64 # batch size
lr = 0.001  # learning rate lr = 0.0002
# divide data into two subsets: Train & Test, of which the test set is the last "days_test" days
days_test = 3
len_test = 140
tw = 1


class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X
def creat_dataset(dataset, tw):
    data_x = []
    data_y = []
    for i in range(len(dataset)-tw):
        data_x.append(dataset[i:i+tw])
        data_y.append(dataset[i+tw])
    return np.asarray(data_x), np.asarray(data_y)  # 转为ndarray数据

def main():
    # load data
    print("loading data...")
    # x_data_all = []
    # y_data_all = []
    dataframe = pd.read_csv('zgpa_train.csv',
                            header=0, parse_dates=[0],
                            index_col=0, usecols=[0, 5], squeeze=True)
    print(dataframe)
    # dataset = dataframe.values
    # data = pd.read_csv('zgpa_train.csv', header=None, usecols=[5], dtype='int')
    dataset = dataframe.values
    print('dataset:', dataset)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))
    print(dataset.shape)
    len_all = dataset.shape[0]
    x_data_train = dataset[:-len_test]
    x_data_test = dataset[-len_test:]
    x_train, y_train = creat_dataset(x_data_train, tw)
    x_test, y_test = creat_dataset(x_data_test, tw)


    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(input_dim=50, output_dim=100, return_sequences=True)))
    # model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(input_dim=100, output_dim=200, return_sequences=True)))
    # model.add(Dropout(0.2))

    model.add(LSTM(300, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(200))
    model.add(Dense(output_dim=1))

    model.add(Activation('relu'))
    start = time.time()
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    print("training model...")
    history = model.fit(x_train, y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        verbose=2)
    score = model.evaluate(
        x_test, y_test, batch_size=64, verbose=0)
    model.save('Bilstm.h5')

    # model.save()
    predict = model.predict(x_test, batch_size=64)
    predict = scaler.inverse_transform(predict)
    y_test = scaler.inverse_transform(y_test)
    rmse = math.sqrt(mean_squared_error(y_test, predict))
    print('specific rmse = ', rmse)
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, 'b', label='real')
    plt.plot(predict, ls='-.', c='r', label='predict')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('kk.png')
    plt.show()
if __name__ == '__main__':
    main()
