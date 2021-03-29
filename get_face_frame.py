# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:51:37 2020

@author: fsb
"""

'''
face info: /media/nas2/Deepfakedeetectionchallenge/tfrecords/all_frames/dfdc_train_part_XX/*.csv
video_info: /

data agumentation:
1, extract face region:
    a, input a face location info, return a facial region mask(fill polygon in opencv)
    b, (optional) implement gaussian filter to facial mask to get a new mask.
     
2, rescale the face twice. 1st is enlarger->ensmaller. 2nd is ensmaller->enlarger.
3, align new face according to the mask to the original larger bounding box.

'''
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy.ma as ma
import pandas


def extract_face_region(w, h, x_leye, y_leye, x_reye, y_reye, x_lmouth, y_lmouth, x_rmouth, y_rmouth):
    # 1, get a polygon
    # 2, gaaussian the mask
    
    def get_k_size(x_leye, x_reye):
        length = int(abs(x_leye - x_reye)/2)
        length = length + length % 2 + 1
        return length
    
    mask = np.zeros((h, w))
    polygon = np.array([[x_leye, y_leye], [x_reye, y_reye], [x_rmouth, y_rmouth], [x_lmouth, y_lmouth]], np.int32)
    cv2.fillConvexPoly(mask, polygon, 1)
    kernel_size = get_k_size(x_leye, x_reye)
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 10)
    mask = mask > 0.1
    #ar = np.nonzero(ar)
    return np.dstack((mask,mask,mask))

def get_coord(begin_pt, eyes_array, mouths_array):
    x_leye = eyes_array[0] - begin_pt[0]
    y_leye = eyes_array[1] - begin_pt[1]
    x_reye = eyes_array[2] - begin_pt[0]
    y_reye = eyes_array[3] - begin_pt[1]
    
    x_lm = mouths_array[0] - begin_pt[0]
    y_lm = mouths_array[1] - begin_pt[1]
    x_rm = mouths_array[2] - begin_pt[0]
    y_rm = mouths_array[3] - begin_pt[1]
    return x_leye, y_leye, x_reye, y_reye, x_lm, y_lm, x_rm, y_rm

def get_face_frame(video_name, csv_file, num_frame_output=2, resize_size=96, blur=0, max_min_switch=0):
    
    face_info = pandas.read_csv(csv_file)
    face_loc = face_info.values
    
    cap = cv2.VideoCapture(video_name)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    org_lap_list = []
    out_frame = []
    
    if(max_min_switch==0):
        for ii in range(num_frame_output):
            out_frame.append((1e10, np.zeros((3,3,3))))
    else:
        for ii in range(num_frame_output):
            out_frame.append((0, np.zeros((3,3,3))))
    
    for ii in range(min(n_frame, face_loc.shape[0])):
        ret, org = cap.read()
        bounding_box = face_loc[ii, 4:8]
        #print(bounding_box)
        bounding_box[0] -= (bounding_box[3]/2)
        bounding_box[1] -= (bounding_box[2]/2)
        bounding_box[2] *= 2.
        bounding_box[3] *= 2.
        bounding_box = bounding_box.astype(np.int32)
        #print(bounding_box)
        eyes = face_loc[ii, 8:12]
        mouth = face_loc[ii, 14:18]
        
        org = org[bounding_box[1] : bounding_box[1] + bounding_box[3], \
                  bounding_box[0] : bounding_box[0] + bounding_box[2],...]
        bounding_box[2] = org.shape[1]
        bounding_box[3] = org.shape[0]
        if(bounding_box[1] < 0 or bounding_box[0] < 0 or org.shape.count(0)>0):
            continue
        #org = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x_leye, y_leye, x_reye, y_reye, x_lm, y_lm, x_rm, y_rm = get_coord(bounding_box[0:2], eyes, mouth)
        
        # generated some additional fake frames from real frames.
        if(resize_size!=0):
            mask = extract_face_region(bounding_box[2], bounding_box[3], \
                                       x_leye, y_leye, x_reye, y_reye, x_lm, y_lm, x_rm, y_rm)
            #face = oeg * np.dstack((mask,mask,mask))
            face = cv2.resize(org, (resize_size, resize_size))
            if(blur==1):
                face = cv2.GaussianBlur(face, (5, 5), 1)
            face = cv2.resize(face, (bounding_box[2], bounding_box[3]))
            face = org * np.logical_not(mask) + face * mask
        else:
            face = org
        face.astype(np.int8)
        
        org_lap = cv2.Laplacian(org, -1)
        org_dense = np.sum(org_lap)
        org_lap_list.append(np.sum(org_lap))
        
        '''
        plt.figure()
        plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        plt.show()
        '''
        #get largest/smallest dense frame
        # min
        if(max_min_switch==0):
            if (org_dense < out_frame[-1][0]):
                out_frame[-1] = (org_dense, face)    
        # max
        else:
            if (org_dense > out_frame[0][0]):
                out_frame[0] = (org_dense, face)
        out_frame.sort(key=lambda x: x[0])
    cap.release()
    return out_frame, org_lap_list            

'''
video = 'jdkvgcjxje.mp4'
csv = 'jdkvgcjxje.csv'
frames, lap_list = get_face_frame(video_name=video, csv_file=csv, max_min_switch=1)

for data in frames:
    plt.figure()
    plt.imshow(cv2.cvtColor(data[1], cv2.COLOR_BGR2RGB))
    plt.show()
''' 

