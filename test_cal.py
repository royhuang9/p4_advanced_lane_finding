#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:52:01 2017

@author: roy
"""

from PIL import Image
import numpy as np

import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from tools import natural_keys


with open('./cal_data.pk', 'rb') as fp:
    data = pickle.load(fp)
    
camera_mtx = data['cal']
dist_coeff = data['dist']
new_camera_mtx = data['new']

file_msk = './camera_cal/calibration*.jpg'
allfiles = glob(file_msk)

allfiles.sort(key=natural_keys)

for filename in allfiles:
    #test code for individial image
    image = np.asarray(Image.open(filename).convert('L'))
    
    dst_img = cv2.undistort(image,camera_mtx, dist_coeff, None, new_camera_mtx)
    
    dst_pil = Image.fromarray(np.uint8(dst_img))
    savefile = filename.replace('camera_cal', 'output_images')
    dst_pil.save(savefile)

    plt.figure(figsize=(12,9))
    plt.gray()
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title(filename)
    
    plt.subplot(1,2,2)
    plt.imshow(dst_img)
    plt.title(filename)
    
file_msk = './test_images/straight_lines*.jpg'
allfiles = glob(file_msk)
allfiles.sort(key = natural_keys)
# for the two straight line image, shrink the width into half, keep height
for filename in allfiles:
    image = np.asarray(Image.open(filename).convert('L'))
    dst_img = cv2.undistort(image,camera_mtx, dist_coeff, None, new_camera_mtx)
    
    dst_pil = Image.fromarray(np.uint8(dst_img))
    savefile = filename.replace('test_images', 'output_images')
    dst_pil.save(savefile)

    plt.figure(figsize=(12,9))
    plt.gray()
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title(filename)
    
    plt.subplot(1,2,2)
    plt.imshow(dst_img)
    plt.title(filename)