# Advanced Lane Finding

## Introduction
Finding the lane is important for the car to know where to drive.  Although P1 already provided some experiments on lane finding, the techniques used in P1 ware canny  and hough transformation by which we can only detect straight lines.

In this project, advanced techniques, like camera calibration, perspective transformation, sliding window searching, sobel filter and HSV color space, are used to detect lane more accuracilly.

<center>![Lane fond](out_images/lane_found.png)</center>

## Camera Calibration
Camera calibration is to get the distortion parameters and perspective transforation matrix.
With the common cheap pinhole camera, there is usually significant distortion. OpenCV takes into account the radial and tangential distortion. 

A image is took by projecting 3D points into image plane using a perspective transformation which is composed with the camera mtraix, or the matrix of instrisc parameters and extrinsic parameters. The intrinsic parameters are determined by the camera, but the extrinsic parameters are also affected by world coordinates.

OpenCV is used to get distortion coeffients and camera matrix. The black-white chessboard method is used. Firstly cv2.findChessboard function can find all the corners of chessboard in image and cv2.cornerSubPix will help fine the coordinates. Secondly cv2.calibrateCamera will get the distortion coeffient and camera matrix.
<center>![Camera calibration](out_images/camera_undist.png)</center>
