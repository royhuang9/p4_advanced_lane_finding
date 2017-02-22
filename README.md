# Advanced Lane Finding

## Introduction
Finding the lane is important for the car to know where to drive.  Although P1 already provided some experiments on lane finding, the techniques used in P1 ware canny  and hough transformation by which we can only detect straight lines.

In this project, advanced techniques, like camera calibration, perspective transformation, sliding window searching, sobel filter and HSV color space, are used to detect lane more accuracilly.

### insert pictures of result.

## Camera Calibration
Camera calibration is to get the distortion parameters and perspective transforation matrix.
With the common cheap pinhole camera, there is usually significant distortion. OpenCV takes into account the radial and tangential distortion. 
$$
x_{corrected}=x(1+k_1r^2+k_2r^4+k_3r^6)\\
y_{corrected}=y(1+k_1r^2+k_2r^4+k_3r^6)
$$

Tangential distortion is corrected by the formulas:
$$
x_{corrected}=x+[2p_1xy+p_2(r^2+2x^2)]\\
y_{corrected}=y+[p_1(r^2+2y^2)+2p_2xy]
$$
So OpenCV provided five distortion parameters in default:
$$
Distortion_{coefficients}=(k_1\;k_2\;p_1\;p_2\;k_3)
$$

A image is took by projecting 3D points into image plane using a perspective transformation.
$$
S\begin{bmatrix}
\mu\\
\nu\\
1
\end{bmatrix}=
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X\\
Y\\
Z\\
\end{bmatrix}
$$
In the right side of the forumula, the first part is the camera mtraix, or the matrix of instrisc parameters. The second part is the extrinsic parameters. The intrinsic parameters are determined by the camera, but the extrinsic parameters are also affected by world coordinates.

OpenCV is used to get distortion coeffients and camera matrix. The black-white chessboard method is used. First cv2.findChessboard function can find all the corners of chessboard in image and cv2.cornerSubPix will help fine the coordinates. Second cv2.calibrateCamera will get the distortion coeffient and camera matrix.
<center>![Camera calibration](out_images/camera_undist.png)</center>
