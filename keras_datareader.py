
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
from keras_vggface import utils

# keras build a graph to output a feature sequence
NUM_TRAIN_PTS = 31666
NUM_TEST_PTS = 3590
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DECAY = 1e-5
EPOCHES = 10

def data_generator(train_test_switch = 0, folder_name='/media/nas2/Shengbang/DFDC_CNN_LSTM_DATA/'):
    def load_data(frame_info):
        file_path = frame_info[0]
        label = frame_info[1]
        npy = np.load(file_path, mmap_mode='r')
        frame = npy[0][38:262,38:262,:]
        frame /= 127.5
        frame -= 1.
        return(frame.astype(np.float16), label)
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
            file_list.append((folder_name + folder + '/' + train_or_test + '_real/' + train, 0))
        for train in train_fake:
            file_list.append((folder_name + folder + '/' + train_or_test + '_fake/' + train, 1))
    random.shuffle(file_list)
    npy_list = []
    label_list = []
    for files in tqdm.tqdm(file_list[:15833]):
        data = load_data(files)
        npy_list.append(data[0])
        label_list.append(data[1])

    output_npy = np.array(npy_list)
    output_label = np.array(label_list)
    np.save('/media/nas2/Shengbang/npy/single_31666_faces_'+train_or_test+'_1.npy', output_npy)
    np.save('/media/nas2/Shengbang/npy/single_31666_faces_label_'+train_or_test+'_1.npy', output_label)
    
    npy_list.clear()
    label_list.clear()
    del output_npy, output_label
    for files in tqdm.tqdm(file_list[15833:]):
        data = load_data(files)
        npy_list.append(data[0])
        label_list.append(data[1])

    output_npy = np.array(npy_list)
    output_label = np.array(label_list)
    np.save('/media/nas2/Shengbang/npy/single_31666_faces_'+train_or_test+'_2.npy', output_npy)
    np.save('/media/nas2/Shengbang/npy/single_31666_faces_label_'+train_or_test+'_2.npy', output_label)

data_generator()
data_generator(train_test_switch = 1)

