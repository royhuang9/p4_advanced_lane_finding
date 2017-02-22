#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:52:01 2017

@author: roy

Calibrate the camera and store the parameter in file

"""

from PIL import Image
import numpy as np

import cv2
from glob import glob
#import matplotlib.pyplot as plt
import pickle

from tools import natural_keys

#the number of inside corners in y
nx = 9
#the number of inside corners in y 
ny = 6

img_msk = './camera_cal/calibration*.jpg'
all_imgnames = glob(img_msk)

all_imgnames.sort(key=natural_keys)
print(all_imgnames)

img_points = []
obj_points = []

objp = np.zeros((ny*nx, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for imgname in all_imgnames:
    image = np.asarray(Image.open(imgname).convert('L'))
    
    ret, corners = cv2.findChessboardCorners(image, (nx, ny), None)
    if ret == True:
        obj_points.append(objp)
        
        cv2.cornerSubPix(image, corners,(11,11),(-1,-1),criteria)
        img_points.append(corners)
        
        '''
        cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
        plt.figure()
        plt.gray()
        plt.imshow(image)
        '''
        
ret, camera_mtx, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
                            obj_points, img_points, image.shape[::-1], None, None,
                            flags=cv2.CALIB_TILTED_MODEL)

image = np.asarray(Image.open('./camera_cal/calibration1.jpg').convert('L'))
h,  w = image.shape[:2]
new_camera_mtx, roi=cv2.getOptimalNewCameraMatrix(camera_mtx, dist_coeff, (w,h) , 0, (w,h))

print('camera_mtx', camera_mtx)
print('dist_coeff', dist_coeff)
print('new_camera_mtx', new_camera_mtx)

'''
print('rvecs', rvecs)
print('tvecs', tvecs)
'''
calfile = './cal_data.pk'
data = {'cal':camera_mtx,
        'dist':dist_coeff,
        'new':new_camera_mtx}

with open(calfile, 'wb') as fp:
    pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)