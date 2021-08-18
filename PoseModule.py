import cv2
import numpy as np
import mediapipe as mp
import time
import math #angle을 구하기위해서 사용

#클래스 선언
class poseDetector():

    def __init__(self, mode=False, upBody=False,smooth=True,
                 detectionCon=0.5, trackingCon=0.5): #파이썬 클래스, false를 주어 빠른 감지를함
        # static_image_mode = False,
        # upper_body_only=False,
        # min_detection_confidence=0.5,
        # min_tracking_confidence=0.5):
        self.mode=mode
        self.upBody=upBody
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackingCon=trackingCon


        self.mpDraw=mp.solutions.drawing_utils
        #포즈를 감지하여 여기에 쓸것임.
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.upBody,self.smooth, self.detectionCon
                                   ,self.trackingCon)

        #포즈찾기
    def findPose(self, img, index ,draw=True ): #사용자는 그림을 그리시겠습니까 아니면 이미지에 표시하겠습니까
        black_img = np.zeros((640, 480, 3), dtype=np.uint8)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # print(results.pose_landmarks) #결과를 확인 x,y,z좌표 랜드마크를 확인
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
                self.mpDraw.draw_landmarks(black_img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
                cv2.imwrite('Result/image%d.jpg' % index, black_img)
                index = index + 1
                #이미지의 좌표에 점을 생성,라인생성
                #내가 원하는 랜드마크 번호 5를 원하면

        return img


    def findPosition(self,img,draw=True):
        self.lmList=[]
        if self.results.pose_landmarks: #결과가 사용가능한 경우
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                # print(id,lm)
                cx,cy=int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                   cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
                    # print(id, cx,cy)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True): #각도 구하는 함수, p1, p2 ,p3를 구해서 각도 구한다

        #Get the landmarks
        # == _, x1, y1 = self.lmLIst[p1]
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        #calculate the angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2)-math.atan2(y1-y2, x1-x2))

        if angle < 0:
            angle += 360
        # print(angle)

        #Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255),3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255),3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)),(x2 - 50, y2 + 50), #x2를 기준으로 밑에다가 angle값을 써줌
                        cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2) #문자열로 바꿔야지 받아들여짐

        return angle

def main():
    cap = cv2.VideoCapture('1.mp4')
    pTime = 0
    detector=poseDetector()


    while True:
        success, img = cap.read()
        index=0
        img=detector.findPose(img,index)
        lmList=detector.findPosition(img,draw=False) #draw를 false시켜서 값만 가져옴 밑에서 그릴 수 있음
        print(lmList[14]) #14는 미디어파이프의 데이터셋에서 가져온 팔꿈치의 위치가 14번이라
        #14번의 좌표값만 보여줌 !
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED) #14번 팔꿈치만 크게 만듬

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)  # 프레임을 줄이기위해서 10이면 초당 50 60프레임이 된다. 하지만 모델을 사용하면 그것에 대해 걱정할 필요가 없다.
    #모듈을 만들기 위해 필요한 과정
if __name__ == "__main__":
    main()