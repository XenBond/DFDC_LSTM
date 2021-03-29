#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:24:25 2020

@author: shengbang
"""
import numpy as np
import random

def data_generator_test(batch_size=0, train_test_switch=0):
    while True:
        yield ([np.ones((16, 28, 28)), np.ones((16,28,28))], np.ones((16,)))

'''
test = data_generator_test()
print(next(test)[0][0].shape)
'''    

def data_generator(batch_size= 32, train_test_switch = 0, folder_name='/media/nas2/Shengbang/npy/'):
    TRAINING_PTS = 31666
    TESTING_PTS = 3590
    def get_label(y1, y2):
    # same: 1 diff: 0
        if (y1==y2):
            return 1
        else:
            return 0
    filenames = []
    labels_list = []
    if(train_test_switch == 0):
        filenames.append('single_31666_faces_train_1.npy')
        filenames.append('single_31666_faces_train_2.npy')
        labels_list.append('single_31666_faces_label_train_1.npy')
        labels_list.append('single_31666_faces_label_train_2.npy')
        
        data_1 = np.load(folder_name + filenames[0], mmap_mode='r')
        data_2 = np.load(folder_name + filenames[1], mmap_mode='r')
        label_1 = np.load(folder_name + labels_list[0], mmap_mode='r')
        label_2 = np.load(folder_name + labels_list[1], mmap_mode='r')
        
        idx_list = list(range(TRAINING_PTS))
        random.shuffle(idx_list)
        
        iteration = len(filenames)
        ii = 0
        while True:
            pairs = []
            for idx in range(batch_size):
                pairs.append([idx_list[ii + idx], idx_list[ii + idx + 1]])
            ii += batch_size
            output = []
            labels = []
            for pair in pairs:
                if(pair[0]>15833 - 1):
                    if(pair[1]>15833 - 1):
                        pair_1 = data_2[pair[0] - 15833]
                        pair_2 = data_2[pair[1] - 15833]
                        label_pair = get_label(label_2[pair[0] - 15833], label_2[pair[1] - 15833])
                    else: 
                        pair_1 = data_2[pair[0] - 15833]
                        pair_2 = data_1[pair[1]]
                        label_pair = get_label(label_2[pair[0] - 15833], label_1[pair[1]])
                else:
                    if(pair[1]>15833 - 1):
                        pair_1 = data_1[pair[0]]
                        pair_2 = data_2[pair[1] - 15833]
                        label_pair = get_label(label_1[pair[0]], label_2[pair[1] - 15833])
                    else:
                        pair_1 = data_1[pair[0]]
                        pair_2 = data_1[pair[1]]
                        label_pair = get_label(label_1[pair[0]], label_1[pair[1]])
                output.append([pair_1[38:262, 38:262, ...], pair_2[38:262, 38:262, ...]])
                labels.append(label_pair)
            output_batch = np.transpose(np.array(output), (1, 0, 2, 3, 4))
            yield ([output_batch[0], output_batch[1]], np.array(labels))
        
            if ((ii + batch_size + 1) > iteration):
                random.shuffle(idx_list)
                random.shuffle(idx_list)
                ii = 0
                        
    else:
        filenames.append('single_3590_faces_test_1.npy')
        labels_list.append('single_3590_faces_label_test_1.npy')
        data_1 = np.load(folder_name + filenames[0], mmap_mode='r')
        label_1 = np.load(folder_name + labels_list[0], mmap_mode='r')
        
        idx_list = list(range(TESTING_PTS))
        random.shuffle(idx_list)
        
        iteration = len(filenames)
        ii = 0
        while True:
            pairs = []
            for idx in range(batch_size):
                pairs.append([idx_list[ii + idx], idx_list[ii + idx + 1]])
            ii += batch_size
            output = []
            labels = []
            for pair in pairs:
                output.append([data_1[pair[0]][38:262, 38:262, ...], data_1[pair[1]][38:262, 38:262, ...]])
                labels.append(get_label(label_1[pair[0]], label_1[pair[1]]))
            output_batch = np.transpose(np.array(output), (1, 0, 2, 3, 4))
            yield ([output_batch[0], output_batch[1]], np.array(labels))
            if ((ii + batch_size) > iteration):
                random.shuffle(idx_list)
                random.shuffle(idx_list)
                ii = 0
'''
pair_data = data_generator(train_test_switch=1)
for ii in range(1):
    a = next(pair_data)[1]
    print(a.shape)
'''

        
'''
dataset = np.load(folder_name + filename)
labels = np.load(folder_name + label_filename)
labels = one_hot(labels, 2)
length = dataset.shape[0]
iteration = int(length / batch_size)
ii = 0
while True:
    yield (dataset[ii * batch_size : (ii+1) * batch_size, ...], labels[ii * batch_size : (ii+1) * batch_size,0,:])
    ii = (ii + 1) % iteration
'''
