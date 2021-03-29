#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:24:21 2020

@author: shengbang
"""


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import random
import os

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

'''
lstm_model = Sequential()
lstm_model.add(Dense(512, input_dim=2048))
lstm_model.add(Dropout(0.5))
lstm_model.add(Activation('tanh'))
lstm_model.add(Dense(2, activation='softmax'))

Adam_opt = Adam(lr=0.001)
lstm_model.compile(optimizer=Adam_opt,
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
model.fit(x_train, y_train)
'''

def build_dataset():
    train_real = '/media/nas2/Shengbang/DFDC_CNN_LSTM_DATA/dfdc_train_part_00/train_real/'
    train_fake = '/media/nas2/Shengbang/DFDC_CNN_LSTM_DATA/dfdc_train_part_00/train_fake/'
    test_real = '/media/nas2/Shengbang/DFDC_CNN_LSTM_DATA/dfdc_train_part_00/test_real/'
    test_fake = '/media/nas2/Shengbang/DFDC_CNN_LSTM_DATA/dfdc_train_part_00/test_fake/'
    
    real_list = os.listdir(train_real)
    fake_list = os.listdir(train_fake)
    test_list_real = os.listdir(test_real)
    test_list_fake = os.listdir(test_fake)
    
    train_list = []
    for real in real_list:
        train_list.append((train_real + real, 0))
    for fake in fake_list:
        train_list.append((train_fake + fake, 1))
        
    test_list = []
    for real in test_list_real:
        test_list.append((test_real + real, 0))
    for fake in test_list_fake:
        test_list.append((test_fake + fake, 1))
    
    random.shuffle(train_list)
    return train_list, test_list


class Inception_Module:
    def __init__(self):
        self.module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3", trainable=False)

    def get_input_from_file(self, filename, label):
        label = label
        input_example = np.load(filename)
        input_example = input_example / 255.0
        features = self.module(input_example)
        return features, label

def data_generator():

train_list, test_list = build_dataset()
Inception_module = Inception_Module()
for ii in range(10):
    x, y = Inception_module.get_input_from_file(train_list[ii][0], train_list[ii][1])
    print(x, y)