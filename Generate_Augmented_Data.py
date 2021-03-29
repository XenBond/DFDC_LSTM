# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import os
import random
import tqdm
import get_face_frame
import matplotlib.pyplot as plt
import cv2
import numpy as np
import gc

class File_Reader:
    def __init__(self):
        self._trainValList_folder_path = '/media/nas2/Deepfakedeetectionchallenge/pre-proc/tf-faces/'
        self._trainValList_folder = os.listdir(self._trainValList_folder_path)
        self._output_folder_path = '/media/nas2/Shengbang/DFDC_SELECTED_AGUMENTED_FRAMES'
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

face_info_dir = '/media/nas2/Deepfakedeetectionchallenge/tfrecords/all_frames'
file_reader = File_Reader()
file_reader.create_folder()
path = '/media/nas2/Shengbang/DFDC_SELECTED_AGUMENTED_FRAMES/'
#file_reader.get_real_fake_number()
resize_list = [64, 96, 128, 160]
for part_num in tqdm.tqdm(range(40, 45)):
    file_reader.get_trainValList(part_num)
    
    train_list_real = file_reader.get_current_train_list_real()
    train_list_fake = file_reader.get_current_train_list_fake()
    for ii, filename in enumerate(train_list_real):
        # fake list should deal independently
        real_name = train_list_real[ii].split('/')[-1]
        folder = train_list_real[ii].split('/')[-2]
        csv_real = '/'.join([face_info_dir, folder, real_name[:-4] + '.csv'])
        if(os.path.exists(csv_real)==False):
            continue
               
        real_frames_min_vague, real_lap_list = get_face_frame.get_face_frame(train_list_real[ii], csv_real, num_frame_output=16, resize_size=0, blur=0, max_min_switch=0)
        real_frames_max_vague, _ = get_face_frame.get_face_frame(train_list_real[ii], csv_real, num_frame_output=16, resize_size=0, blur=0, max_min_switch=1)
        
        if(real_frames_min_vague is not None):
            np.save(path + train_list_real[ii].split('/')[-2] + '/train_real/' + real_name[:-4] + '_min.npy', np.array(real_frames_min_vague))
        if(real_frames_max_vague is not None):
            np.save(path + train_list_real[ii].split('/')[-2] + '/train_real/' + real_name[:-4] + '_max.npy', np.array(real_frames_max_vague))
        if(real_lap_list is not None):
            np.save(path + train_list_real[ii].split('/')[-2] + '/train_real/' + real_name[:-4] + '_lap_sequence.npy', np.array(real_lap_list))
        print('finish 2 types of real!')
        
        resize_size = random.choice(resize_list)
        fake_frames_resize_64_min, _ = get_face_frame.get_face_frame(train_list_real[ii], csv_real, num_frame_output=2, resize_size=resize_size, blur=0, max_min_switch=0)
        fake_frames_resize_64_max, _ = get_face_frame.get_face_frame(train_list_real[ii], csv_real, num_frame_output=2, resize_size=resize_size, blur=0, max_min_switch=1)
        fake_frames_resize_64_min_blur, _ = get_face_frame.get_face_frame(train_list_real[ii], csv_real, num_frame_output=2, resize_size=resize_size, blur=1, max_min_switch=0)
        fake_frames_resize_64_max_blur, _ = get_face_frame.get_face_frame(train_list_real[ii], csv_real, num_frame_output=2, resize_size=resize_size, blur=1, max_min_switch=1)
        
        if(fake_frames_resize_64_min is not None):
            np.save(path + train_list_real[ii].split('/')[-2] + '/train_fake/' + real_name[:-4] + '_resize_'+str(resize_size)+'_min.npy', np.array(fake_frames_resize_64_min))
        if(fake_frames_resize_64_max is not None):
            np.save(path + train_list_real[ii].split('/')[-2] + '/train_fake/' + real_name[:-4] + '_resize_'+str(resize_size)+'_max.npy', np.array(fake_frames_resize_64_max))
        if(fake_frames_resize_64_min_blur is not None):
            np.save(path + train_list_real[ii].split('/')[-2] + '/train_fake/' + real_name[:-4] + '_resize_'+str(resize_size)+'_min_blur.npy', np.array(fake_frames_resize_64_min_blur))
        if(fake_frames_resize_64_max_blur is not None):
            np.save(path + train_list_real[ii].split('/')[-2] + '/train_fake/' + real_name[:-4] + '_resize_'+str(resize_size)+'_max_blur.npy', np.array(fake_frames_resize_64_max_blur))
        print('finish 4 types of 64 resize!')
        gc.collect()
        
    for ii, filename in enumerate(train_list_fake):
        fake_name = train_list_fake[ii].split('/')[-1]
        folder = train_list_fake[ii].split('/')[-2]
        csv_fake = '/'.join([face_info_dir, folder, fake_name[:-4] + '.csv'])
        if(os.path.exists(csv_fake)==False):
            continue
        fake_frames_min_vague, fake_lap_list = get_face_frame.get_face_frame(train_list_fake[ii], csv_fake, num_frame_output=2, resize_size=0, blur=0, max_min_switch=0)
        fake_frames_max_vague, _ = get_face_frame.get_face_frame(train_list_fake[ii], csv_fake, num_frame_output=2, resize_size=0, blur=0, max_min_switch=1)
        if(fake_frames_min_vague is not None):
            np.save(path + train_list_fake[ii].split('/')[-2] + '/train_fake/' + fake_name[:-4] + '_min.npy', np.array(fake_frames_min_vague))
        if(fake_frames_max_vague is not None):
            np.save(path + train_list_fake[ii].split('/')[-2] + '/train_fake/' + fake_name[:-4] + '_max.npy', np.array(fake_frames_max_vague))
        if(fake_lap_list is not None):
            np.save(path + train_list_fake[ii].split('/')[-2] + '/train_fake/' + fake_name[:-4] + '_lap_sequence.npy', np.array(fake_lap_list))
        print('finish 2 types of fake!')
        gc.collect()
        
    test_list_real = file_reader.get_current_test_list_real()
    test_list_fake = file_reader.get_current_test_list_fake()
    for ii, filename in enumerate(test_list_real):
        real_name = test_list_real[ii].split('/')[-1]
        fake_name = test_list_fake[ii].split('/')[-1]
        folder_real = test_list_real[ii].split('/')[-2]
        folder_fake = test_list_fake[ii].split('/')[-2]
        csv_real = '/'.join([face_info_dir, folder_real, real_name[:-4] + '.csv'])
        csv_fake = '/'.join([face_info_dir, folder_fake, fake_name[:-4] + '.csv'])
        if(os.path.exists(csv_real)==False or os.path.exists(csv_fake)==False):
            continue
        real_frames_min_vague, real_lap_list = get_face_frame.get_face_frame(train_list_real[ii], csv_real, num_frame_output=16, resize_size=0, blur=0, max_min_switch=0)
        real_frames_max_vague, _ = get_face_frame.get_face_frame(train_list_real[ii], csv_real, num_frame_output=16, resize_size=0, blur=0, max_min_switch=1)
        fake_frames_min_vague, fake_lap_list = get_face_frame.get_face_frame(train_list_fake[ii], csv_fake, num_frame_output=16, resize_size=0, blur=0, max_min_switch=0)
        fake_frames_max_vague, _ = get_face_frame.get_face_frame(train_list_fake[ii], csv_fake, num_frame_output=16, resize_size=0, blur=0, max_min_switch=1)
        
        if(real_frames_min_vague is not None):
            np.save(path + test_list_real[ii].split('/')[-2] + '/test_real/' + real_name[:-4] + '_min.npy', np.array(real_frames_min_vague))
        if(real_frames_max_vague is not None):
            np.save(path + test_list_real[ii].split('/')[-2] + '/test_real/' + real_name[:-4] + '_max.npy', np.array(real_frames_max_vague))
        if(fake_frames_min_vague is not None):
            np.save(path + test_list_fake[ii].split('/')[-2] + '/test_fake/' + fake_name[:-4] + '_min.npy', np.array(fake_frames_min_vague))
        if(fake_frames_max_vague is not None):
            np.save(path + test_list_fake[ii].split('/')[-2] + '/test_fake/' + fake_name[:-4] + '_max.npy', np.array(fake_frames_max_vague))
        
        if(real_lap_list is not None):
            np.save(path + test_list_real[ii].split('/')[-2] + '/test_real/' + real_name[:-4] + '_lap_sequence.npy', np.array(real_lap_list))
        if(fake_lap_list is not None):
            np.save(path + test_list_fake[ii].split('/')[-2] + '/test_fake/' + fake_name[:-4] + '_lap_sequence.npy', np.array(fake_lap_list))
        
        print('finish 2 types of test!')
        gc.collect()
        
