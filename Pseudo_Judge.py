from math import inf
import cv2
import numpy as np
import time
import PoseModule as pm
from utils.defineLabel import defineLabel
from utils.WriteCSV import WriteCSV


# 모델 불러오기
# Read(Model)


# 영상 선택 및 실행
#cap=cv2.VideoCapture(0) #카메라 번호

pre_label = 0
cur_label = 0
isdown = True
iswrong = False

while(1):
    
    # Mediapipe 수행
    # 스켈레돈을 통해 각 키포인트 좌표와 각도 구함
    # 그리진 말기ㅇㅇ 밑에서 틀린부분 있으면 색칠해야하니까
     

    # 해당 프레임을 학습모델에 넣어보고 결과 레이블링값을 cur_label에 대입
    # cur_label=MODEL(frame)
    if(pre_label != cur_label):
        if(isdown):
            if(pre_label == 3):
                isdown = False
                iswrong = False
            elif(cur_label == pre_label):
                print('유지 중')
                iswrong = False
            elif(cur_label == (pre_label + 1)):
                pre_label += 1
                iswrong = False
            else:
                print('wrong')
                iswrong = True
                # 각도로 어떤곳이 잘못되었는지 판단
        else:
            if(pre_label == 1):
                isdown = True
                iswrong = False
            elif(cur_label == pre_label):
                print('유지 중')
                iswrong = False
            elif(cur_label == (pre_label - 1)):
                pre_label -= 1
                iswrong = False
            else:
                print('wrong')
                iswrong = True
                # 각도로 어떤곳이 잘못되었는지 판단
    
    if(iswrong):
        print('잘못된 자세입니다!')
        # 어디가 잘못되었는지 알려주기
        # def defineLabel(elbow_angle, hip_angle, knee_angle, frame, start_sec, end_sec) 을 이용하면 될듯?
        # 함수 자체를 일부 수정해서 쓰거나, 코드 부분만 채용해서 쓰기



        
    # 아 그럼 스켈레톤으로 그림그리는 부분은 여기다가 해야할듯? (빨간점으로 해야하니까)