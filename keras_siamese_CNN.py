#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:38:19 2020

@author: shengbang
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras_vggface.vggface import VGGFace
import keras_siamese_reader
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, \
                        Lambda, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras import backend as K
import keras

num_classes = 2
epochs = 200

NUM_TRAIN_PTS = 31666
NUM_TEST_PTS = 3590
BATCH_SIZE = 24

def euclidean_distance(vects):
    x, y = vects
    
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    # cos distance
    
    '''
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

    '''
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.5
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    '''
    base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
    for layer in base_model.layers[-2:]:
        layer.trainable=True
    for layer in base_model.layers[:-2]:
        layer.trainable=False
    return base_model
    '''
    # 299
    #base_model = VGGFace(model='vgg16', include_top=False, input_shape=input_shape)
    base_model = VGGFace(model='vgg16', include_top=False, input_shape=(224,224,3))
    for layer in base_model.layers:
        layer.trainable=False
    last_layer = base_model.output
    x = Flatten()(last_layer)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    last_layer = Dense(1, activation='sigmoid')(x)
    model = Model(base_model.input, last_layer)
    model.load_weights('_weights.01-0.44.hdf5')
    return Model(model.input, model.get_layer('dense_2').output)
    '''
    with open('facenet_senet50.json', 'r') as f:    
        model = keras.models.model_from_json(f.read())
        model.load_weights('_weights.01-1.11.hdf5')
        for layer in model.layers[-3:]:
            layer.trainable = True
        last_layer = model.get_layer('dense_2').output
        return Model(model.input, last_layer)
    '''


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < .5, y_true.dtype)))

K.clear_session()
# network definition
base_network = create_base_network((224, 224, 3))

input_a = Input(shape=(224, 224, 3))
input_b = Input(shape=(224, 224, 3))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)

train_generator = keras_siamese_reader.data_generator(batch_size=BATCH_SIZE)
test_generator = keras_siamese_reader.data_generator(batch_size=BATCH_SIZE, train_test_switch=1)
# train
cp_callback = keras.callbacks.ModelCheckpoint(filepath='./_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
tb_callback = keras.callbacks.TensorBoard(log_dir='./tensorboard',
                                             update_freq=1000)
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,restore_best_weights=True)
train_callbacks = [cp_callback,tb_callback,es_callback]

model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
model.fit_generator(train_generator, steps_per_epoch=int(NUM_TRAIN_PTS / BATCH_SIZE),
                    epochs=epochs, validation_data=test_generator, callbacks=train_callbacks, \
                    validation_steps=int(NUM_TEST_PTS / BATCH_SIZE), shuffle=True)
'''
model.fit_generator(train_generator, steps_per_epoch=int(num_train_pts / 32), \
          epochs=100, validation_data=test_generator, \
          validation_steps=int(num_test_pts / 32), shuffle=True)
'''
K.clear_session()
# compute final accuracy on training and test sets
#y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
#tr_acc = compute_accuracy(tr_y, y_pred)
#y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
#te_acc = compute_accuracy(te_y, y_pred)

#print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))