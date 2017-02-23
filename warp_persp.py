#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:35:19 2017

@author: roy
"""
from PIL import Image
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

src = np.asarray([[524, 500], [768, 500], [910, 600], [392, 600]], dtype=np.float32)
dst = np.asarray([[196, 400], [455, 400], [455, 600], [196, 600]], dtype=np.float32)

mt_persp=cv2.getPerspectiveTransform(src, dst)


# read calibration data
with open('./cal_data.pk', 'rb') as fp:
    data = pickle.load(fp)
    
camera_mtx = data['cal']
dist_coeff = data['dist']
new_camera_mtx = data['new']

file_name = './test_images/straight_lines1.jpg'

image = np.asarray(Image.open(file_name).convert('L'))
img_undist = cv2.undistort(image, camera_mtx, dist_coeff, None, new_camera_mtx)

img_size = (image.shape[1]//2, image.shape[0])

data = {'mt':mt_persp, 'img_size':img_size}
warp_data_file = './warp_data.pk'
with open(warp_data_file, 'wb') as fp:
    pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

img_h, img_w = image.shape

img_warped = cv2.warpPerspective(img_undist, mt_persp, img_size)


fig = plt.figure(figsize=(12,9))
gs = gridspec.GridSpec(1, 2,
                       width_ratios=[2,1],
                       height_ratios=[1,1]
                       )

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

ax1.imshow(img_undist, cmap='Greys_r')
ax1.plot(src[:,0], src[:,1], 'ro')
ax1.axis('off')
ax1.set_title('Straight Line 1, undistorted')
ax1.set_xlim(0, image.shape[1])
ax1.set_ylim(image.shape[0], 0)

ax2.imshow(img_warped)
ax2.plot(dst[:,0], dst[:,1],'ro')
ax2.axis('off')
ax2.set_title('Bird view')
ax2.set_xlim(0, img_size[0])
ax2.set_ylim(img_size[1], 0)

fig.tight_layout()