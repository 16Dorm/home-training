import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5 ):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils #손에 점을 그리기 위해 필요

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLns in self.results.multi_hand_landmarks:  # landmark에 xy좌표와 id 번호가 있다. 이미 올바른 순서대로 나열 되어있
                if draw:
                    self.mpDraw.draw_landmarks(img, handLns,self.mpHands.HAND_CONNECTIONS)  # mpHands.HAND_CONNECTIONS 점을 이어줌,#handLns 손에 빨간 점 생성
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList=[]
        if self.results.multi_hand_landmarks:
            #우리가 말하는 손을 적어야함.
            myHand=self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):  # id와 xy좌표 얻기
                # print(id,lm)#id, xy좌표 프린트
                h, w, c = img.shape  # 높이 너비 채널 이것은 우리에게 너비와 높이를 준다.
                cx, cy = int(lm.x * w), int(lm.y * h)  # 여기에 쓸 수 있도록 위치를 찾을 수 있다는 것 정수로 바꿈 볼수있도록
                # print(id, cx, cy)  # 손의 좌표를 x와 y로 뽑아냄
                lmList.append([id,cx,cy])
                if draw:
                # if id == 0:  # 첫번째 랜드마크에 대해 이야기하는것임
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)  # 첫번째인 점의 위치를 7의 크기의 보라색 원으로 표현

        return lmList


def main():
    # 프레임과 시간 체크
    pTime = 0  # 이전시간
    cTime = 0  # 현재시간
    cap = cv2.VideoCapture(0) #카메라 번호
    detector=handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[4]) #4번 점의 위치를 기준으로 xy좌표를 표시해줌.

        # 프레임 구하기
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()