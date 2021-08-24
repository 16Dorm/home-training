import cv2
import mediapipe as mp
import time
import numpy as np
mpDraw=mp.solutions.drawing_utils
#포즈를 감지하여 여기에 쓸것임.
mpPose=mp.solutions.pose
pose=mpPose.Pose()


f = open("video_name.txt", 'r')
m_name = f.read()
cap = cv2.VideoCapture(m_name + ".mp4")

index = 0

pTime=0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    black_img = np.zeros((640, 480, 3), dtype=np.uint8)
    # print(results.pose_landmarks) #결과를 확인 x,y,z좌표 랜드마크를 확인
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS) #이미지의 좌표에 점을 생성,라인생성
        mpDraw.draw_landmarks(black_img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        cv2.imwrite('Result/image%d.jpg'%index,black_img)
        index=index+1
        #내가 원하는 랜드마크 번호 5를 원하면
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy=int(lm.x*w), int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
            print(id, cx,cy)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1) #프레임을 줄이기위해서 10이면 초당 50 60프레임이 된다. 하지만 모델을 사용하면 그것에 대해 걱정할 필요가 없다.



