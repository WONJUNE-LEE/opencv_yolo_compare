import torch
import matplotlib.pyplot as plt
import numpy

model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5m, yolov5x, custom

#이미지를 불러오고 욜로모델에다가 넣는 함수, A function that loads an image and puts it into the YOLO
def resultShow(imgname):
    results = model(imgname)

    results.show()

    return results

left = resultShow('calibration_complete\cupL_100_23.png') #왼쪽 결과, left result
right = resultShow('calibration_complete\cupR_100_23.png') #오른쪽 결과, right result 

print(left.xyxy[0]) #yolo에서 인식한 결과 출력, Output the result recognized by yolo
print(right.xyxy[0]) #yolo에서 인식한 결과 출력, Output the result recognized by yolo

#인식한 물체중에 내가 거리측정을 원하는 객체를 찾아야함, We need to find the object We want to measure distance from among the recognized objects.
#disparity = left.xyxy[0][0][0] - right.xyxy[0][0][0] #cupLR_60_8, cupLR_27_8, cupLR2_27_8, chair_117_8
disparity = left.xyxy[0][1][0] - right.xyxy[0][1][0] #cupLR_180_8
#disparity = left.xyxy[0][2][0] - right.xyxy[0][1][0]
disparity = disparity.cpu().detach().numpy() #cuda to float

print('두 이미지 간의 차이', disparity, 'pixel') #print disparity

focal_length = 5.4 #카메라마다 다름, focas_length
camera_between = 230#카메라 간의 사이, distance between camera

distance = (focal_length*camera_between/disparity) * 13.71

print('물체와의 거리', distance, 'cm')