import numpy as np
import cv2
from matplotlib import pyplot as plt


#좌우 스테레오 이미지 불러오기, Import left and right stereo images
imgL = cv2.imread('assets\cupL2_27_8.jpg')
imgR = cv2.imread('assets\cupR2_27_8.jpg')


#스테레오SGBM 함수 파라미터 입력 및 저장, Enter and save stereoSGBM function parameters
window_size = 3
min_disp = 0
num_disp = 130 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 1,
    disp12MaxDiff = -1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2
)

#함수호출 -> 깊이맵 생성, Function call -> Create depth map
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0 

#깊이맵 출력, print depth map
plt.imshow(disparity, 'gray')
plt.show()

#거리측정을 원하는 부분의 좌표선택, 
# Select the coordinates of the part you want to measure the distance
#print(disparity[427][694]) #여기서는 컵부분을 선택, Select the part that is cup
print(disparity[434][891])

pixel_disparity = disparity[434][891]
focal_length = 5.4 #카메라 초점거리
camera_between = 80 #카메라 사이거리

distance = (focal_length*camera_between/pixel_disparity) * 8.08
print('opencv로 계산한 물체 사이의 거리', distance, 'cm')




