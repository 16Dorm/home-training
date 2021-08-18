import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("1.mp4")
# cap = cv2.VideoCapture(0) #카메라 번호
# cap2=cv2.VideoCapture(0) #카메라 번호

detector =pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img =cap.read()
    if not success:
        break
    img = cv2.resize(img, (1280,720)) #영상의 크기 조절, 프레임 조절할 수 있다
    # img = cv2.resize(img, (1920, 1080))
    # img = cv2.imread("2.PNG")  # 각도를 얻기 위한 이미지 각도를 얻고 주석
    # 이후에 할일은 포즈 모듈을 가져와야함 포즈 모듈로 각도 재기
    img = detector.findPose(img, False) #false를 해서 우리가 보고자하는 점 외에는 다 제거
    lmList = detector.findPosition(img, False) #그리기를 원하지 않으므로 false
    # print(lmList) #좌표를 프린트
    if len(lmList)!=0:
        # print(lmList[24][1])  # 24번은 엉덩이 x축 좌표만
        # print(lmList[1][1]) #1번은 눈을 표시 x축 좌표만
        # Right Arm
        if lmList[24][1]<lmList[1][1]:
            angle = detector.findAngle(img, 12,14,16)
            # 각도를 퍼센트로 나타내는 코드
            per = np.interp(angle, (65, 160), (100, 0))
            print(angle, per)
            bar = np.interp(angle, (65, 160), (650, 100))  # 앞에가 최소 뒤에가 최대
        # Left Arm
        else:
            angle=detector.findAngle(img, 11, 13, 15)
            # 각도를 퍼센트로 나타내는 코드
            per = np.interp(angle, (195, 265), (100, 0))
            # print(angle, per)
            bar = np.interp(angle, (195, 265), (650, 100))  # 앞에가 최소 뒤에가 최대

        # #각도를 퍼센트로 나타내는 코드
        # angle = detector.findAngle(img, 11, 13, 15)
        # per = np.interp(angle,(65,160),(100,0))
        # # print(angle, per)
        # bar = np.interp(angle, (65, 160), (650 , 100)) #앞에가 최소 뒤에가 최대

        #check for the push up curls
        color = (255,0,255)
        if per==100:
            color = (0, 255, 0)
            if dir ==0: #올라가고있다.
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir==1:
                count += 0.5
                dir = 0
        # print(count)


        #draw bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 4)

        #draw curl count
        cv2.rectangle(img, (0,450), (250,720),(0,255,0),cv2.FILLED)
        cv2.putText(img, str(int(count)), (45,670), cv2.FONT_HERSHEY_PLAIN, 15,
        (255, 0, 0) ,25)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)
    cv2.imshow("Image",img)
    cv2.waitKey(1)