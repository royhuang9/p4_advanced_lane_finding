#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:48:26 2017

@author: roy
"""

import pickle

from moviepy.editor import VideoFileClip

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



mylane = Lane(nwindows=20, margin=30, minpix=20)


def process_image(image):

    dst = pipeline(image, camera_mtx, dist_coeff, new_camera_mtx, persp_mt, persp_size, mylane)
    return dst
    
project_output = 'project_out.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(project_output, audio=False)