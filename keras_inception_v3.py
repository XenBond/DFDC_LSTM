

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:07:38 2020

@author: shengbang
"""
from keras.layers import GlobalAveragePooling2D, LSTM, TimeDistributed
from keras.layers import Dense, Dropout, Activation, Input, Flatten, BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.models import model_from_json
import keras

from keras_vggface.vggface import VGGFace
import tqdm
import tensorflow as tf 
import os
import numpy as np
import random

# keras build a graph to output a feature sequence
num_train_pts = 31666 * 40
num_test_pts = 3590 * 40
#num_train_pts = 34522
#num_test_pts = 3590
BATCH_SIZE =48

K.clear_session()

def data_generator(train_test_switch = 0, folder_name='/media/nas2/Shengbang/DFDC_CNN_LSTM_DATA/', batch_size=BATCH_SIZE):
    def load_data(frame_info):
        file_path = frame_info[0]
        frame_idx = frame_info[1]
        label = frame_info[2]
        npy = np.load(file_path, mmap_mode='r')
        frame = npy[frame_idx][38:262,38:262,:] / 127.5 - 1
        del npy
        return(frame, label)
    file_list = []
    folder_list = os.listdir(folder_name)
    if train_test_switch == 0:
        train_or_test = 'train'
    else:
        train_or_test = 'test'
    for folder in folder_list:
        train_real = os.listdir(folder_name + folder + '/' + train_or_test + '_real/')
        train_fake = os.listdir(folder_name + folder + '/' + train_or_test + '_fake/')
        for train in train_real:
            for frame_idx in range(40):
                file_list.append((folder_name + folder + '/' + train_or_test + '_real/' + train, frame_idx, 0))
        for train in train_fake:
            for frame_idx in range(40):
                file_list.append((folder_name + folder + '/' + train_or_test + '_fake/' + train, frame_idx, 1))
    random.shuffle(file_list)

    # batch output
    length = len(file_list)
    iteration = int(length / batch_size)
    ii = 0
    while True:
        random.shuffle(file_list)
        batch_list = []
        label_list = []
        file_batch = file_list[ii * batch_size : (ii+1) * batch_size]
        for files in file_batch:
            data = load_data(files)
            batch_list.append(data[0])
            label_list.append(data[1])
        yield (np.array(batch_list), np.array(label_list))
        ii = (ii + 1) % iteration
        batch_list.clear()
        label_list.clear()
        file_batch.clear()

train_generator = data_generator(batch_size=BATCH_SIZE)
test_generator = data_generator(train_test_switch=1, batch_size=BATCH_SIZE)

K.clear_session()
    
base_model = VGGFace(model='vgg16', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable=False
last_layer = base_model.output
x = Flatten(name='flatten')(last_layer)
x = Dense(512, activation='relu', name='fc6')(x)
x = Dense(512, activation='relu', name='fc7')(x)
output = Dense(1, activation='sigmoid', name='fc8')(x)
model = Model(base_model.input, output)
model.load_weights('_weights.01-0.44.hdf5')
optimizer_ = Adam(learning_rate=1e-5, decay=1e-6)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer_,
              metrics=['accuracy'])

# train
cp_callback = keras.callbacks.ModelCheckpoint(filepath='./_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
tb_callback = keras.callbacks.TensorBoard(log_dir='./tensorboard',
                                             update_freq=1000)
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,restore_best_weights=True)
train_callbacks = [cp_callback,tb_callback,es_callback]
'''
model.fit_generator(train_generator, steps_per_epoch=int(num_train_pts / BATCH_SIZE / 40), \
          epochs=200, validation_data=test_generator, callbacks=train_callbacks,\
          validation_steps=int(num_test_pts / BATCH_SIZE / 40), shuffle=True)
'''
loss, acc = model.evaluate_generator(test_generator, steps=int(num_test_pts / BATCH_SIZE))
print('loss: ',loss,' acc: ', acc)
# Serialize weights to JSON

model_json = model.to_json()
with open('facenet_vgg16.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights("facenet_vgg16.h5")

K.clear_session()

        
