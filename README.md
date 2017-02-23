# Advanced Lane Finding

## Introduction
Finding the lane is important for the car to know where to drive.  Although P1 already provided some experiments on lane finding, the techniques used in P1 ware canny  and hough transformation by which we can only detect straight lines.

In this project, advanced techniques, like camera calibration, perspective transformation, sliding window searching, sobel filter and HSV color space, are used to detect lane more accuracilly. The following is an example.

<center>![Lane fond](out_images/lane_found.png)</center>

## Camera Calibration
Camera calibration is to get the distortion parameters and perspective transforation matrix.
With the common cheap pinhole camera, there is usually significant distortion. OpenCV takes into account the radial and tangential distortion. 

A image is took by projecting 3D points into image plane by a perspective transformation which is composed with the camera mtraix, or the matrix of instrisc parameters and extrinsic parameters. The intrinsic parameters are determined by the camera, but the extrinsic parameters are also affected by world coordinates.

OpenCV is used to get distortion coeffients and camera matrix. The black-white chessboard method is used. Firstly cv2.findChessboard function can find all the corners of chessboard in image and cv2.cornerSubPix will help fine the coordinates. Secondly cv2.calibrateCamera will get the distortion coeffient and camera matrix. Thirdly cv2.getOptimalNewCameraMatrix with alpha set to zero to get a new camera matrix. All the three part parameters are stored in a file for later use.

<center>![Camera calibration](out_images/camera_undist.png)</center>

##Pipeline
###Distortion correction
Already got the camera matrix and distortion parameters. Then every captured frame should be transformed to compensate distortion. The following are the code and frames before and after undistorted. They are not apprent like the chessboard, but the diffence can be got by substracting them.
```
img_undist = cv2.undistort(image, camera_mtx, dist_coeff, None, new_camera_mtx)
```
<center>![Road undistorted sample](out_images/road_undistort.png)</center>

###Binary image generation
It is time to appy color threshold and sobel filter to generate a binary image which includes lane pixel. 

The following is the filter functions.
```python
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
```
Most filters are just copied from Udacity. But the sat_filter is changed to overcome the shadow of tree. In fact sat_filter combine the staturation filter and brightness filter. 
```
def sat_filter(image, s_threshold=(120,255)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    #clear the higher part half of image v_channel which is sky
    img_h, img_w = v_channel.shape
    v_channel[:img_h//2, :] = 0
    
    # rescale with the lower part
    v_channel = (255 * v_channel/np.max(v_channel))

    #combine them together
        
    combined = np.zeros_like(s_channel)
    combined[ ((s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1]) & \
             (v_channel > 125.5)) ] = 1
    
    return combined
```
For example, the following is only the saturation channel throldholded with [120, 255]. The back of black car and shadow of tree are also included.
<center>![thresholded saturation channel](out_images/shadow1.png)</center>
After apply brithness threshold on value channel and combine it with thresholded saturation channel, only the yellow line is left:
<center>![sat and value](out_images/shadow2.png)</center>

Then combine staturation, magnitude, direction and sobel thresholded, get a final binary image.
```
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
```
The visualized binary image is following:
<center>![combined binary image](out_images/combined.png)</center>

###Perspective transform
In order to find lane on a binary image, it is better to transform it to a bird-eye view. The straight_lines1.jpg is undistored first. Four points are selected manually on left and right lines as source points, then define four new points in bird-eye view as destination. The four points in the bird-eye view form rectangle corners. cv2.getPerspectiveTransform is called to calculate the perspective transform matrix. One of the advantage of bird-eye view is most unrelated part in image disappeared, then it is easy to search the lanes.
<center>![Bird-eye view](out_images/warp_persp.png)</center>
The perspective matrix can be used to warp other images, like below:
<center>![Bird-eye view](out_images/warp_test2.png)</center>
###Find lanes
Although the lane is very obvious in the bird-eye view, but we need to find the pixel location of the lane. The method to find lane for the first frame and following frame are different. The blind search method is choosed for the first frame.
####First frame
Take a histogram of the half part of the binary image to find the start point of lane. The peak location of the left half is the left lane starting point, and the right half is the same.
<center>![histogram](out_images/hist1.png)</center>
```python
        img_h, img_w = image.shape
        window_height = img_h//self.nwindows
        
        hstg = np.sum(image[img_h//2:, :], axis=0)
  
        midpoint = img_w//2
        leftx_base = np.argmax(hstg[:midpoint])
        rightx_base = np.argmax(hstg[midpoint:]) + midpoint
```
Once we found the starting point of the lane, a search window in size 60x36 is placed for each lane. The starting point is at the middle of the bottom of the search window. All the pixels in the search window is recorded. The histogram of pixels in window is calculated. The new peak location is the middle of the bottom of the next search window. Repeat the process until the whole image is searched and all pixels in the series of search window is recored. It looks like this:
<center>![search windows](out_images/search_windows.png)</center>
With all the pixels found in all the search window, we can call numpy.polyfit function to fit a best quadratical polynomial which is draw in green line above. Of cource, left and right lane has its indepedent polynomial.
After get the lanes and polynomial, we can paint the road the tranfrom back into the orignal frame. 
The code for this algorithm is from Udacity. I am not going to copy it here.
###Curvature and distance
The curvature of lane is calculated in term of the formula. The value of x and y should be converted from pixel to measurement in meter. The following is fucntion calculating the curvature of the lane. The radius is the reciprocal of curvature.
```python
    def cal_curva(self, py):
        fit_rw = np.polyfit(self.ally*self.mppy, self.allx*self.mppx, 2)
        py_rw = py * self.mppy
        curva = (2.0*fit_rw[0]) / ((1.0 + (2*fit_rw[0]*py_rw+fit_rw[1])**2)**1.5)
        return curva
```
About the position of the vehicle, we can calculate the offset between the middle of two lanes and the center of the image,

The final result is below:
<center>![final result](out_images/final.png)</center>
##Project Video


Videos below show performance of our algorithm on project and challenge videos.

[![Project Video](https://youtu.be/9rEWE1zmgro)

##Discussion
I tried my algorithm with the challenge video and harder challenge. I have to say it is totally a disaster. The problem faced is the parallel inference line with lane which is difficult to be removed. The sharp turn can also be hard to deal with. Bad weather condition is also challenge, like raining, snowing and night. No much idea how to overcome these issues yet.