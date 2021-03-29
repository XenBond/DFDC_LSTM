#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:54:52 2020

@author: shengbang
"""

import cv2
import File_Reader
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import tensorflow as tf
from mtcnn import MTCNN
#import time
import tqdm
import sys
import gc

class Video_Reader:
    def __init__(self):
        self._mtcnn_file = '/media/nas2/Deepfakedeetectionchallenge/mtcnn-0.1.0-py3-none-any.whl'
        self._THRESHOLD = 0.9
        self._interpolate_FLAG = 1
        self._detector = MTCNN()
        self._output_path = '/media/nas2/Shengbang/DFDC_CNN_LSTM_DATA/'
           
    def extract_frames_from_video(self, filename, num_frame, display=False):
        pass

    def extract_faces_from_video(self, filename, interpolate_scale, num_frame, face_size=150, display=False, return_faces=True):
        # input: filename, interpolate scale you want, number of frame you want.
        # output: frame_list, output bounding box.
        #start_time = time.process_time()
        if(interpolate_scale!=1):
            self._interpolate_FLAG = interpolate_scale
        if(filename[-4:]!= '.mp4'):
            raise ValueError("Input must be a mp4 file.")
        cap = cv2.VideoCapture(filename)
        x_list = []
        y_list = []
        w_list = []
        h_list = []
        frame_list = []
        if(cap.isOpened()==False):
            return 0,0,0,0,0
        for ii in range(num_frame):          
            ret, old_frame = cap.read()
            if(ret == False):
                return 0,0,0,0,0
            frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
            if(ii % int(num_frame / interpolate_scale) == 0 or ii == (num_frame - 1)):
                faces = self._detector.detect_faces(frame)      
                
                # Two situation that we don't want this video
                if (faces == []):
                    print('no face found')
                    #end_time = time.process_time()
                    #print('Time: ', end_time - start_time)
                    return 0,0,0,0,0
                sorted(faces, key=lambda i: i['confidence'], reverse=True)
                if faces[0]['confidence'] < self._THRESHOLD:
                    print('Fail to find a face')
                    #end_time = time.process_time()
                    #print('Time: ', end_time - start_time)
                    return 0,0,0,0,0
                
                x,y,w,h=faces[0]['box']
                x_list.append(x + w/2)
                y_list.append(y + h/2)
                w_list.append(w)
                h_list.append(h)
        cap.release()
        output_x = list(range(num_frame))
        output_y = list(range(num_frame))
        output_x = self.cubic_interpolation(x_list, num_frame)
        output_y = self.cubic_interpolation(y_list, num_frame)
        w = np.mean(w_list)
        h = np.mean(h_list)
              
        # Display faces:      
        if(return_faces==True):
            face_list = []
            for ii, frame in enumerate(frame_list):
                x = output_x[ii]
                y = output_y[ii]
                #face_list.append(frame[int(y-face_size/2):int(y+face_size/2), int(x-face_size/2):int(x+face_size/2)])
                face_reg = frame[int(y-h/2):int(y+h/2), int(x-h/2):int(x+h/2)]
            
                if(y<h/2 or x<h/2 or face_reg.shape[0]!=face_reg.shape[1] or face_reg.shape.count(0)>0):
                    print("ROI too big.", face_reg.shape)
                    return 0,0,0,0,0
                #face_reg = cv2.cvtColor(face_reg, cv2.COLOR_RGB2GRAY)
                face_reg = cv2.resize(face_reg, (299, 299))
                face_list.append(face_reg)
            
            if(display==True):
                print('face region size: w:',w,'h:',h)
                for face in face_list:
                    plt.figure()
                    plt.imshow(face)
                    plt.show()
                    plt.close()
            #end_time = time.process_time()
            #print('Time: ', end_time - start_time)
            return face_list, output_x, output_y, w, h
        else:            
            if(display==True):
                print('face region size: w:',w,'h:',h)
                for ii, frame in enumerate(frame_list):
                    x = output_x[ii]
                    y = output_y[ii]
                    cv2.rectangle(frame, (x-int(w/2), y-int(h/2)), (x+int(w/2), y+int(h/2)), (255, 255, 255), 10)    
                    plt.figure()
                    plt.imshow(frame)
                    plt.show()
                    plt.close()
            #end_time = time.process_time()
            #print('Time: ', end_time - start_time)
            return frame_list, output_x, output_y, w, h
        
    def cubic_interpolation(self, input_list, length, show_spline=False):
        # get scale info from output list.
        input_array = np.array(input_list)
        scale = int(length / (len(input_list) - 1))
        x = np.array([i * scale for i in range(len(input_list))])
        b_n = interpolate.make_interp_spline(x, input_array)
        output_list = []
        for ii in range(length):
            output_list.append(int(b_n(ii)))
        if(show_spline==True):
            plt.plot([i * scale for i in range(len(input_list))], input_list)
            plt.plot(range(len(output_list)), output_list)
            plt.show()
        return output_list
        
    def set_face_detector_threshold(self, threshold):
        self._THRESHOLD = threshold
        
    def get_output_path(self):
        return self._output_path


video_reader = Video_Reader()
file_reader = File_Reader.File_Reader()
file_reader.create_folder()
#file_reader.get_real_fake_number()
path = video_reader.get_output_path()
for part_num in tqdm.tqdm(range(50)):
    file_reader.get_trainValList(part_num)  
    train_list_real = file_reader.get_current_train_list_real()
    train_list_fake = file_reader.get_current_train_list_fake()
    for ii, filename in enumerate(train_list_real):
        real_name = train_list_real[ii].split('/')[-1]
        fake_name = train_list_fake[ii].split('/')[-1]
        real_face_list, _, _, _, _ = video_reader.extract_faces_from_video(train_list_real[ii], 5, 40)
        fake_face_list, _, _, _, _ = video_reader.extract_faces_from_video(train_list_fake[ii], 5, 40)
        if(real_face_list==0 or fake_face_list==0):
            continue
        output_real = np.array(real_face_list)
        output_fake = np.array(fake_face_list)
        np.save(path + train_list_real[ii].split('/')[-2] + '/train_real/' + real_name[:-4], output_real)
        np.save(path + train_list_fake[ii].split('/')[-2] + '/train_fake/' + fake_name[:-4], output_fake)
        
        del real_name, fake_name, real_face_list, fake_face_list, output_real, output_fake
        gc.collect()
        
    test_list_real = file_reader.get_current_test_list_real()
    test_list_fake = file_reader.get_current_test_list_fake()
    for ii, filename in enumerate(test_list_real):
        real_name = test_list_real[ii].split('/')[-1]
        fake_name = test_list_fake[ii].split('/')[-1]
        real_face_list, _, _, _, _ = video_reader.extract_faces_from_video(test_list_real[ii], 5, 40)
        fake_face_list, _, _, _, _ = video_reader.extract_faces_from_video(test_list_fake[ii], 5, 40)
        if(real_face_list==0 or fake_face_list==0):
            continue
        output_real = np.array(real_face_list)
        output_fake = np.array(fake_face_list)
        np.save(path + test_list_real[ii].split('/')[-2] + '/test_real/' + real_name[:-4], output_real)
        np.save(path + test_list_fake[ii].split('/')[-2] + '/test_fake/' + fake_name[:-4], output_fake)
        
        del real_name, fake_name, real_face_list, fake_face_list, output_real, output_fake
        gc.collect()

    del train_list_real, train_list_fake, test_list_real, test_list_fake
    gc.collect()
