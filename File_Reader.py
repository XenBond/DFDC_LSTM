# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import os
import random

class File_Reader:
    def __init__(self):
        self._trainValList_folder_path = '/media/nas2/Deepfakedeetectionchallenge/pre-proc/tf-faces/'
        self._trainValList_folder = os.listdir(self._trainValList_folder_path)
        self._output_folder_path = '/media/nas2/Shengbang/DFDC_CNN_LSTM_DATA'
        self._part_folder_list = list(range(0,50))
        self._current_part = -1
        self._current_train_list_real = []
        self._current_train_list_fake = []
        self._current_test_list_real = []
        self._current_test_list_fake = []
        
    def get_real_fake_number(self):
        total_train_real = 0
        total_train_fake = 0
        total_test_real = 0
        total_test_fake = 0
        for part_number, trainValList in enumerate(self._trainValList_folder):
            pickle_path = self._trainValList_folder_path + trainValList + '/trainValList.pkl'
            with open(pickle_path, 'rb') as pickle_file:
                x = pickle.load(pickle_file)
                print(part_number, 'part has:')
                print('for train:')
                print('REAL: ', len(x['train']['REAL']))
                total_train_real += len(x['train']['REAL'])
                print('FAKE: ', len(x['train']['FAKE']))
                total_train_fake += len(x['train']['FAKE'])
                print('for test:')
                print('REAL: ', len(x['test']['REAL']))
                total_test_real += len(x['test']['REAL'])
                print('FAKE: ', len(x['test']['FAKE']))
                total_test_fake += len(x['test']['FAKE'])
        print('total train real video: ', total_train_real)
        print('total train fake video: ', total_train_fake)
        print('total test real video: ', total_test_real)
        print('total test fake video: ', total_test_fake)
                
    def get_trainValList(self, part_number):
         self._current_part = part_number
         pickle_path = self._trainValList_folder_path + self._trainValList_folder[part_number] + '/trainValList.pkl'
         with open(pickle_path, 'rb') as pickle_file:
            x = pickle.load(pickle_file)
            self._current_train_list_real.clear()
            self._current_train_list_fake.clear()
            self._current_test_list_real.clear()
            self._current_test_list_fake.clear()
            self._current_train_list_real = x['train']['REAL']
            self._current_train_list_fake = x['train']['FAKE']
            self._current_test_list_real = x['test']['REAL']
            self._current_test_list_fake = x['test']['FAKE']
            
    def get_current_train_list_real(self):
        return self._current_train_list_real
    
    def get_current_train_list_fake(self):
        return self._current_train_list_fake
    
    def get_current_test_list_real(self):
        return self._current_test_list_real
    
    def get_current_test_list_fake(self):
        return self._current_test_list_fake

    def get_current_part_number(self):
        return self._current_part
    
    def create_folder(self):
        for folder in self._trainValList_folder:
            os.makedirs('/'.join([self._output_folder_path, folder, 'train_real']), exist_ok=True)
            os.makedirs('/'.join([self._output_folder_path, folder, 'train_fake']), exist_ok=True)
            os.makedirs('/'.join([self._output_folder_path, folder, 'test_real']), exist_ok=True)
            os.makedirs('/'.join([self._output_folder_path, folder, 'test_fake']), exist_ok=True)

        
'''
file_reader = File_Reader()
file_reader.create_folder()
'''