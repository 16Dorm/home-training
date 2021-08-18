import cv2
import mediapipe as mp
import time
import Handtrackingmodule as htm


# 프레임과 시간 체크
pTime = 0  # 이전시간
cTime = 0  # 현재시간
cap = cv2.VideoCapture(0) #카메라 번호
detector=htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False) #draw=False를 사용하면 우리가 모듈에서 설정한 것이 사라짐.
    if len(lmList) !=0:
        print(lmList[4]) #4번 점의 위치를 기준으로 xy좌표를 표시해줌.

    # 프레임 구하기
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


