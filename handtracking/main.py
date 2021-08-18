import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0) #카메라 번호

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw=mp.solutions.drawing_utils #손에 점을 그리기 위해 필요

#프레임과 시간 체크
pTime=0 #이전시간
cTime=0 #현재시간

while True:
  success, img = cap.read()
  imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  results=hands.process(imgRGB)
  # print(results.multi_hand_landmarks)

  if results.multi_hand_landmarks:
    for handLns in results.multi_hand_landmarks: #landmark에 xy좌표와 id 번호가 있다. 이미 올바른 순서대로 나열 되어있다.
      for id, lm in enumerate(handLns.landmark):#id와 xy좌표 얻기
        # print(id,lm)#id, xy좌표 프린트
        h, w, c = img.shape  #높이 너비 채널 이것은 우리에게 너비와 높이를 준다.
        cx, cy = int(lm.x*w), int(lm.y*h)#여기에 쓸 수 있도록 위치를 찾을 수 있다는 것 정수로 바꿈 볼수있도록
        print(id, cx, cy) #손의 좌표를 x와 y로 뽑아냄
        if id==0: #첫번째 랜드마크에 대해 이야기하는것임
          cv2.circle(img, (cx,cy),25,(255,0,255),cv2.FILLED) #첫번째인 점의 위치를 25의 크기의 보라색 원으로 표현


      mpDraw.draw_landmarks(img,handLns,mpHands.HAND_CONNECTIONS)#mpHands.HAND_CONNECTIONS 점을 이어줌,#handLns 손에 빨간 점 생성

#프레임 구하기
  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)




  cv2.imshow("Image", img)
  cv2.waitKey(1)