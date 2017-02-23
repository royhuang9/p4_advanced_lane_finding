#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:59:16 2017

@author: roy
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def sat_filter(image, s_threshold=(120,255)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    #clear the higher part half of image v_channel which is sky
    img_h, img_w = v_channel.shape
    v_channel[:img_h//2, :] = 0
    
    # rescale with the lower part
    v_channel = (255 * v_channel/np.max(v_channel))
    
    combined = np.zeros_like(s_channel)
    combined[((s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1]) & \
             (v_channel > 125.5))] = 1
     
    return combined

    
def sobel_filter(image_gray, orient='x', kern_size = 3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient == 'x':
        sobel_img = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kern_size)
    elif orient == 'y':
        sobel_img = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kern_size)
    
    sobel_img = np.abs(sobel_img)
    scaled_sobel = np.uint8(255 * sobel_img/np.max(sobel_img))
    grad_binary = np.zeros_like(image_gray)
    grad_binary[( scaled_sobel >= thresh[0] ) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_filter(image_gray, kern_size = 3, thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kern_size)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kern_size)
    
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_mag = np.uint8(255 * mag/np.max(mag))
    
    mag_binary = np.zeros_like(sobelx)
    mag_binary[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1
    
    return mag_binary

def direct_filter(image_gray, kern_size = 3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kern_size)
    sobelx = np.abs(sobelx)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kern_size)
    sobely = np.abs(sobely)
    
    direct_value = np.arctan2(sobely, sobelx)
    direct_binary = np.zeros_like(sobelx)
    
    direct_binary[(direct_value > thresh[0]) & (direct_value < thresh[1])] = 1    
    return direct_binary

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img) 
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# undistorted and warp the image
def transform(image, camera_mtx, dist_coeff, new_camera_mtx, persp_mt, persp_size):
    #undist the image
    img_undist = cv2.undistort(image, camera_mtx, dist_coeff, None, new_camera_mtx)
    #image_size = (image.shape[1], image.shape[0])
    img_warped = cv2.warpPerspective(img_undist, persp_mt, persp_size)
    return img_warped
    
def transform_back(image, persp_mt, image_size):
    #print('transform_back shape {}'.format(image.shape))
    #image_size = (image.shape[1], image.shape[0])
    img_unwarped = cv2.warpPerspective(image, persp_mt, image_size,
                flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    return img_unwarped 

import matplotlib.gridspec as gridspec
def show_persp(image1, image2):
        
    fig = plt.figure(figsize=(12,9))
    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[2,1],
                           height_ratios=[1,1]
                           )
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    ax1.imshow(image1, cmap='Greys_r')
    ax1.axis('off')
    ax1.set_title('Binary Image')
    ax1.set_xlim(0, image1.shape[1])
    ax1.set_ylim(image1.shape[0], 0)
    
    ax2.imshow(image2, cmap='Greys_r')
    ax2.axis('off')
    ax2.set_title('Bird view')
    ax2.set_xlim(0, image2.shape[1])
    ax2.set_ylim(image2.shape[0], 0)
    
    fig.tight_layout()
        
def pipeline(image, camera_mtx, dist_coeff, new_camera_mtx, persp_mt, persp_size, lane, file_name=''):
    s_th=(120, 255)
    s_bin = sat_filter(image, s_threshold = s_th)

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    sobel_kern = 5
    sobel_th = (50, 150)
    sobelx_bin = sobel_filter(img_gray, orient='x', kern_size=sobel_kern, thresh=sobel_th)
    sobely_bin = sobel_filter(img_gray, orient='y', kern_size=sobel_kern, thresh=sobel_th)
    
    mag_kern = 9
    mag_th = (50, 255)
    mag_bin = mag_filter(img_gray, kern_size=mag_kern, thresh=mag_th)
    
    dir_kern = 5
    dir_th = (0.7, 1.3)
    direct_bin = direct_filter(img_gray, kern_size=dir_kern, thresh=dir_th)
    
    combined_bin = np.zeros_like(img_gray)
    combined_bin[ (s_bin==1) | ((sobelx_bin==1) & (sobely_bin==1)) \
                 | ((mag_bin==1) & (direct_bin==1))] = 1
    #combined_bin[ (s_bin==1) | ((sobelx_bin==1) & (sobely_bin==1))] = 1
   
    #crop interested region
    
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(imshape[1]//2-50, imshape[0]//2), 
            (imshape[1]//2+50, imshape[0]//2), 
             (imshape[1],imshape[0])]], dtype=np.int32)
    img_reg = region_of_interest(combined_bin, vertices)
    
    #transform image
    img_warped = transform(img_reg, camera_mtx, dist_coeff, new_camera_mtx, persp_mt, persp_size)

    
    #show_persp(combined_bin, img_warped)    
    
    img_lane = lane.find_lane(img_warped)

    #transform back image
    img_unwarped = transform_back(img_lane, persp_mt, (image.shape[1], image.shape[0]))
    
    img_undist = cv2.undistort(image, camera_mtx, dist_coeff, None, new_camera_mtx)
    final_image =cv2.addWeighted(img_undist, 0.7, img_unwarped, 0.3, 0)
    left_radius = 1/lane.left.curva
    right_radius = 1/lane.right.curva
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(final_image, 'Radius, left {:.1f}m right {:.1f}m'\
                .format(left_radius, right_radius), (40,50), font, 1, (0,255,0))
    cv2.putText(final_image, 'Offset from center {:.2f}cm'\
                .format(lane.offset*100), (40,80), font, 1, (0,255,0))
    return final_image


class Line():
    def __init__(self, name='', pty = 660, miss_th=3, count_max=5, weight=[0.2, 0.2, 0.2,0.2,0.2]):
        self.name = name
        self.miss_count = 0
        self.miss_th = miss_th
        self.weight=weight
        
        # was the last detected lane is accepted?
        self.detected = False

        #polynomial coefficients for the most recent fit
        self.fit_params = None
        self.fits = []

        #radius of curvature of the line in some units
        # curvature = 1/radius
        self.curva = 0
        self.curvas = []
        self.count_max = count_max

        #distance in meters of vehicle center from the line
        self.line_base_pos = None
                
        #x and y values for detected line pixels
        self.allx = None
        self.ally = None
        
        #self.mppx = 3.7/540
        self.mppx = 3.7/270
        self.mppy = 3.0/95
        
        self.curve_pty = pty
        self.distance = 0
        
    def cal_curva(self, py):
        fit_rw = np.polyfit(self.ally*self.mppy, self.allx*self.mppx, 2)
        py_rw = py * self.mppy
        curva = (2.0*fit_rw[0]) / ((1.0 + (2*fit_rw[0]*py_rw+fit_rw[1])**2)**1.5)
        return curva
    
    def cal_distance(self, fit_params, py, img_size):
        x = fit_params[0]*(py**2) + fit_params[1] * py + fit_params[0]
        dist = (x - img_size[1]/2) * self.mppx
        return dist
        
    def check_fit(self, fit_params, curva):
        
        angle = np.arctan2(660, (2*fit_params[0]*660+ fit_params[1])) * 180 /np.pi
        if np.abs(angle - 90.0) > 5:
            return False
        
        if np.abs(fit_params[2] - self.fit_params[2]) > 100:
            return False
        
            
        return True
        
    def fit(self, img_size, x, y):
        py = self.curve_pty
        self.allx = x
        self.ally = y
        fit_params = np.polyfit(y, x, 2)
        
        # calculate radius
        curva = self.cal_curva(py)
        self.distance = 0
        
        ret = False
        
        if self.detected is False or self.check_fit(fit_params, curva):
            self.detected = True
            self.curva = curva
            self.curvas.append(np.abs(curva))
            
            if len(self.curvas) > self.count_max:
                self.curvas.pop(0)
            
            
            self.fits.append(fit_params)
            if len(self.fits) > self.count_max:
                self.fits.pop(0)
            
            
            # calculate the distance 
            px = fit_params[0]*(py**2) + fit_params[1]*py + fit_params[2]
        
            # judge whether the new fit is good to be accept
            self.line_base_pos = (px - img_size[0]/2)*self.mppx
            self.miss_count = 0
            
            ret = True
        else:
            #print('{} there'.format(self.name))
            self.miss_count += 1
            # if miss_count is more than threshold,
            if self.miss_count >= self.miss_th:
                self.detected = False
                ret = False
                self.radius = 0
                self.fit_params = [0,0,0]
            else:
                # the last value is not used, but current value is given up
                # self.curva and self.fit_params keep no change
                self.detected = True
                ret = True
        if ret:
            average = [0,0,0]
            total = 0
            for idx, one_fit in enumerate(reversed(self.fits)):
                total += self.weight[idx]
                
                for it in range(len(one_fit)):
                    average[it] += one_fit[it] * self.weight[idx]
                
            for it in range(len(average)):
                average[it] /= total
            self.fit_params = average
            self.distance = self.cal_distance(self.fit_params, self.curve_pty, img_size)
            #print(total, fit_params, average)
            
        return ret, self.fit_params
    
    def set_fit(self, params, offset):
        curve_ptx = params[0] * (self.curve_pty**2) + \
                params[1] * self.curve_pty + params[2]
        
        curve_ptx += offset
        p3 = curve_ptx - (params[0] * (self.curve_pty**2) + params[1] * self.curve_pty)
        self.fit_params[0] = params[0]
        self.fit_params[1] = params[1]
        self.fit_params[2] = p3
        
    def cur_fit(self):
        return self.current_fit
        
class Lane():
    def __init__(self, nwindows = 10, margin=100, minpix = 50, miss_th = 3, pty = 660):
        self.left = Line('left', pty=pty)
        self.right = Line('right')
        self.need_search = True
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.miss_th = miss_th
        self.radius = 0.
        self.offset = 0
        
    def find_lane(self, image):
        if self.need_search:
            image = self.lane_search(image)
        else:
            image = self.lane_match(image)
        
        if self.left.detected is False or self.right.detected is False:
            self.need_search = True
        else:
            self.need_search = False
        
        if self.left.distance !=0 and self.right.distance != 0:
            self.offset = -(self.left.distance + self.right.distance) * self.left.mppx
        return image
    
    # choose the fit parameters in terms of which has more point thant another
    def adjust_fit(self, left_fit, ret_l, right_fit, ret_r):
        if ret_l is True and ret_r is False:
            self.right.set_fit(left_fit, 300)
        elif ret_l is False and ret_r is True:
            self.left.set_fit(right_fit, -300)
        elif ret_l is False and ret_r is False:
            # use the average value
            pass
        
        
        return self.left.fit_params, self.right.fit_params
        
    def lane_search(self, image):
        img_h, img_w = image.shape
        window_height = img_h//self.nwindows
        
        hstg = np.sum(image[img_h//2:, :], axis=0)
        
        
        midpoint = img_w//2
        leftx_base = np.argmax(hstg[:midpoint])
        rightx_base = np.argmax(hstg[midpoint:]) + midpoint
    
        nonzero = image.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        
        leftx_current = leftx_base
        rightx_current = rightx_base
                                
        left_lane_inds = []
        right_lane_inds = []
        
        img_out = np.dstack((np.zeros_like(image), np.zeros_like(image), \
                             np.zeros_like(image)))*255
        for window in range(self.nwindows):
            win_y_low = img_h - (window + 1)*window_height
            win_y_high = img_h - window * window_height
            
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
        
            #cv2.rectangle(img_out, (win_xleft_low, win_y_low), 
            #              (win_xleft_high, win_y_high), (0, 255, 0), 2)
            #cv2.rectangle(img_out, (win_xright_low, win_y_low), 
            #              (win_xright_high, win_y_high), (0, 255, 0), 2)
        
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
        
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        ret_l, left_fit = self.left.fit((img_w, img_h), leftx, lefty)
        
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        ret_r, right_fit = self.right.fit((img_w, img_h), rightx, righty)
        
        if ret_l is False or ret_r is False:
            left_fit, right_fit = self.adjust_fit(left_fit, ret_l, right_fit, ret_r)
        
        
        # find indice for the whole image
        whole_img = np.indices((img_h, img_w))
        imgx= whole_img[1].flatten()
        imgy = whole_img[0].flatten()
        
        #print('imgx shape {}, imgy shape {}'.format(imgx.shape, imgy.shape))
        # got indice between left lane and right lane
        lane_inds = ((imgx >= (left_fit[0]*(imgy**2) + left_fit[1] * imgy + left_fit[2])) 
                        & (imgx < (right_fit[0]*(imgy**2) + right_fit[1] * imgy + right_fit[2])))
        # paint the lane to green
        
        img_out[imgy[lane_inds], imgx[lane_inds]] = [0, 255, 0]
        
        
        ploty = np.linspace(0, img_out.shape[0]-1, img_out.shape[0])
        plot_leftx = np.int_(left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2])
        plot_leftx = np.clip(plot_leftx, 0, img_out.shape[1] - 1, plot_leftx)
        plot_rightx = np.int_(right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2])
        plot_rightx = np.clip(plot_rightx, 0, img_out.shape[1] -1, plot_rightx)
        
        img_out[np.int_(ploty), plot_leftx] = [0, 255, 0]
        img_out[np.int_(ploty), plot_rightx] = [0, 255, 0]
        
        
        #paint left track to Red, right track to green
        img_out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        img_out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        return img_out
    
    def lane_match(self, image):
        left_fit = self.left.fit_params
        right_fit = self.right.fit_params
        
        img_h, img_w = image.shape
        nonzero = image.nonzero()
        nonzerox = nonzero[1]
        nonzeroy = nonzero[0]
    
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1] * nonzeroy
                       + left_fit[2] - self.margin)) & (nonzerox < (left_fit[0] * (nonzeroy**2) + 
                       left_fit[1] * nonzeroy + left_fit[2] + self.margin)))
    
        right_lane_inds = (((nonzerox > right_fit[0]*(nonzeroy**2) + right_fit[1]* nonzeroy
                        + right_fit[2] - self.margin)) & (nonzerox < (right_fit[0] * (nonzeroy**2) + 
                        right_fit[1] * nonzeroy + right_fit[2] + self.margin)))
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        #left_fit = np.polyfit(lefty, leftx, 2)
        #right_fit = np.polyfit(righty, rightx, 2)
        ret_l, left_fit = self.left.fit((img_w, img_h), leftx, lefty)
        ret_r, right_fit = self.right.fit((img_w, img_h), rightx, righty)
        
        if ret_l is False or ret_r is False:
            left_fit, right_fit = self.adjust_fit(left_fit, ret_l, right_fit, ret_r)
        
        img_out = np.dstack((image, image, image))*255
    
        whole_img = np.indices((img_h, img_w))
        imgx= whole_img[1].flatten()
        imgy = whole_img[0].flatten()
        
        lane_inds = ((imgx >= (left_fit[0]*(imgy**2) + left_fit[1] * imgy + left_fit[2])) 
                        & (imgx < (right_fit[0]*(imgy**2) + right_fit[1] * imgy + right_fit[2])))
        img_out[imgy[lane_inds], imgx[lane_inds]] = [0, 255, 0]
        
        img_out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        img_out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        return img_out
