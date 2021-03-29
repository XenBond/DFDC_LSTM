

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:07:38 2020

@author: shengbang
"""

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, LSTM, TimeDistributed
from keras.layers import Dense, Dropout, Activation, Input, Flatten
from keras.models import Model
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.models import model_from_json

import tqdm
import tensorflow as tf 
import os
import numpy as np
import random

# keras build a graph to output a feature sequence
num_train_pts = 31666
num_test_pts = 3590
#num_train_pts = 34522
#num_test_pts = 3590

def data_generator(batch_size= 32, train_test_switch = 0, folder_name='/media/nas2/Shengbang/DFDC_TIME_SERIES/'):
    def one_hot(data, num_classes):
        batch = data.shape[0]
        num = data.shape[1]
        targets = data.reshape(-1).astype(np.int)
        return np.eye(num_classes)[targets].reshape((batch, num, num_classes)).astype(np.int)
    if(train_test_switch == 0):
        filename = 'face_region_31666.npy'
        label_filename = 'face_region_label_31666.npy'
    else:
        filename = 'face_region_3590.npy'
        label_filename = 'face_region_label_3590.npy'
    dataset = np.load(folder_name + filename)
    labels = np.load(folder_name + label_filename)
    labels = one_hot(labels, 2)
    length = dataset.shape[0]
    iteration = int(length / batch_size)
    ii = 0
    while True:
        yield (dataset[ii * batch_size : (ii+1) * batch_size, ...], labels[ii * batch_size : (ii+1) * batch_size,0,:])
        ii = (ii + 1) % iteration

train_generator = data_generator()
test_generator = data_generator(train_test_switch=1)

K.clear_session()
base_model = InceptionV3(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable=False  
    
model = Sequential()
model.add(LSTM(input_shape=(40, 2048), units=200, dropout=0.3, activation='tanh'))
model.add(Dense(units=200, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(units=200, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(units=2, activation='softmax'))
optimizer_ = Adam(learning_rate=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_,
              metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=int(num_train_pts / 32), \
          epochs=100, validation_data=test_generator, \
          validation_steps=int(num_test_pts / 32), shuffle=True)
score, acc = model.evaluate_generator(test_generator, steps=int(num_test_pts / 32))
print(score, acc)

# Serialize weights to JSON
model_json = model.to_json()
with open('lstm_model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

K.clear_session()


        
