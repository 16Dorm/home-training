import cv2
import time
import PoseModule as pm

f = open("video_name.txt", 'r')
m_name = f.read()
cap = cv2.VideoCapture(m_name + ".mp4")

pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)  # draw를 false시켜서 값만 가져옴 밑에서 그릴 수 있음
    if len(lmList) !=0:
        print(lmList[14])  # 14는 미디어파이프의 데이터셋에서 가져온 팔꿈치의 위치가 14번이라
        # 14번의 좌표값만 보여줌 !
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)  # 14번 팔꿈치만 크게 만듬

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # 프레임을 줄이기위해서 10이면 초당 50 60프레임이 된다. 하지만 모델을 사용하면 그것에 대해 걱정할 필요가 없다.
# 모듈을 만들기 위해 필요한 과정