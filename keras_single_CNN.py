
#from keras.layers import GlobalAveragePooling2D, LSTM, TimeDistributed, Conv2D, MaxPooling2D 
#from keras.layers import Dense, Dropout, Activation, Input, Flatten, BatchNormalization
#from keras.models import Model
#from keras.models import Sequential
#from keras import backend as K
#from keras.optimizers import Adam
#from keras.models import model_from_json

import tqdm
import os
import numpy as np
import random

# keras build a graph to output a feature sequence
NUM_TRAIN_PTS = 31666 * 40
NUM_TEST_PTS = 3590 * 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DECAY = 1e-5
EPOCHES = 10

def data_generator(train_test_switch = 0, folder_name='/media/nas2/Shengbang/DFDC_CNN_LSTM_DATA/', batch_size=32):
    def load_data(frame_info):
        file_path = frame_info[0]
        frame_idx = frame_info[1]
        label = frame_info[2]
        npy = np.load(file_path, mmap_mode='r')
        frame = (npy[frame_idx]-128)/128
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


'''
for ii in range(3):
    print(next(train_generator)[0].shape)
'''

K.clear_session()

# 299
model = Sequential()
model.add(Conv2D(64, 3, activation='relu', padding='same', input_shape=(299,299,3)))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3), strides=(2,2)))

# 149
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3), strides=(2,2)))

# 74
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3), strides=(2,2)))

# 36
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3), strides=(2,2)))

# 17
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())

model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(units=1, activation='sigmoid'))
optimizer_ = Adam(learning_rate=1e-3, decay=1e-5)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer_,
              metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=int(NUM_TRAIN_PTS / BATCH_SIZE), \
          epochs=EPOCHES, validation_data=test_generator, \
          validation_steps=int(NUM_TEST_PTS / BATCH_SIZE), shuffle=True)
score, acc = model.evaluate_generator(test_generator, steps=int(NUM_TEST_PTS / BATCH_SIZE))
print(score, acc)

# Serialize weights to JSON
model_json = model.to_json()
with open('vgg19_modified_model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

K.clear_session()












