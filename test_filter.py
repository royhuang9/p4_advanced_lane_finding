#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:45:01 2017

@author: roy
"""
from PIL import Image
import numpy as np

from glob import glob
import matplotlib.pyplot as plt
import pickle
from tools import natural_keys

from filters import pipeline, Lane


# read calibration data
with open('./cal_data.pk', 'rb') as fp:
    data = pickle.load(fp)
    
camera_mtx = data['cal']
dist_coeff = data['dist']
new_camera_mtx = data['new']

# read perspective matrix data
warp_data_file = './warp_data.pk'
with open(warp_data_file, 'rb') as fp:
    data = pickle.load(fp)
persp_mt = data['mt']
persp_size = data['img_size']

files = glob('./test_images/test2.jpg')

mylane = Lane(nwindows=20, margin=30, minpix=20)

files.sort(key=natural_keys)
for file_name in files:    
    image = np.asarray(Image.open(file_name))
    dst = pipeline(image, camera_mtx, dist_coeff, new_camera_mtx, persp_mt, persp_size, mylane, file_name)
    plt.figure(figsize=(12,9))
    plt.imshow(dst)
    plt.title('Lane identified')