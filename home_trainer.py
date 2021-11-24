from classification_model.SupervisedLearning import classification
from math import inf
import math
import cv2
import numpy as np
import time
import utils.PoseModule as pm
from utils.add_Pictogram import add_Pictogram
from utils.add_Lights import add_Lights
from utils.defineLabel import defineLabel
from utils.Pushup_Counting import Pushup_Counting

class GUI_data():
    def __init__(self):
        # GUI_form
        self.weight = 60
        self.goal_cnt = 3
        self.goal_set = 1
        self.isPressedConfirm = False

        # AI
        self.accuracy = False
        self.full_frames = 0
        self.incorrect_frames = 0

        self.full_frames_total = 0
        self.incorrect_frames_total = 0

        self.cur_set_num = 0

        self.cur_light = 'black'

        # GUI_Timer
        self.interval_sec_per_set = 10

        # GUI_result
        self.want_replay_set = 0
        self.Home = True
        self.isReplay = False
        self.Exit = True

class HomeTrainer():
    def run_pose_estimation(video_name, dataset):
        print(dataset.weight, dataset.goal_cnt, dataset.goal_set)
        print(video_name)
        cap = cv2.VideoCapture("./Video/" + video_name + ".mp4")
        #cap = cv2.VideoCapture(0)
       
        # 동영상으로 저장하기 위한 코드
        w = 1280    #round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = 720     #round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적
        print(w,h, fps)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('./play_results/output_' + str(dataset.cur_set_num) + '.avi', fourcc, fps, (w, h))

        # 사전 준비시간을 label0으로 잘라내기 위한 작업 - 실시간캠이면 삭제.
        with open('video_list.txt', 'r') as infile:
            data = infile.readlines()
            for i in data:
                v_name, start_sec, end_sec = (i.split())
                if (video_name == v_name):
                    start_sec = int(start_sec)
                    end_sec = int(end_sec)
                    if (end_sec == 0):
                        end_sec = math.ceil(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/30)
                    break

        # count를 위한 객체 및 변수
        pushup_instance = Pushup_Counting()
        count = 0
        cur_label = -1

        # 게이지 %(percent) 확인을 위한 변수
        per = 0

        # 프레임 확인을 위한 변수
        pTime = 0
        
        # 스켈레톤 검출 및 라벨링 확인을 위한 변수
        index = 0
        label_list = []
        keypoint_list = []

        # 예측 모델 생성
        class_var=classification()
        class_var.train_csv()

        # (준비된)영상 출력을 위한 변수
        #detector =pm.poseDetector("cam")
        detector =pm.poseDetector(video_name)

        while True:
            
            success, img = cap.read()
            if not success:
                break
            img = cv2.resize(img, (1280,720)) #영상의 크기 조절, 프레임 조절할 수 있다
            black_img = np.zeros((480, 640, 3), dtype=np.uint8) ##데이터 저장용 검은배경 이미지 생성
            # img = cv2.imread("2.PNG")  # 각도를 얻기 위한 이미지 각도를 얻고 주석
            # 이후에 할일은 포즈 모듈을 가져와야함 포즈 모듈로 각도 재기

            img = detector.findPose(img, black_img, index, False) #false를 해서 우리가 보고자하는 점 외에는 다 제거
            index += 1
            lmList = detector.findPosition(img, False) #그리기를 원하지 않으므로 false

            keypoint = [] # 핵심 키포인트를 담을 리스트
            if len(lmList)!=0:
                # print(lmList[24][1])  # 24번은 엉덩이 x축 좌표만
                # print(lmList[1][1]) #1번은 눈을 표시 x축 좌표만
                
                # Right Arm
                if lmList[24][1]<lmList[1][1]:
                    elbow_angle = detector.findAngle(img, 12,14,16)
                    if (elbow_angle > 180):
                        elbow_angle  = 360 - elbow_angle
                    hip_angle= detector.findAngle(img,12,24,26)
                    knee_angle= detector.findAngle(img,24,26,28)

                    # 그리기
                    detector.drawPoint(img, 12, 14, 16, 'elbow', elbow_angle)
                    detector.drawPoint(img, 12, 24, 26, 'hip', hip_angle)
                    detector.drawPoint(img, 24, 26, 28, 'knee', knee_angle)

                    # 각도를 퍼센트로 나타내는 코드
                    per = np.interp(elbow_angle, (80, 160), (100, 0))
                    # print(angle, per)
                    bar = np.interp(elbow_angle, (80, 160), (650, 100))  # 앞에가 최소 뒤에가 최대
                    head = lmList[0][2]
                    shoulder = (lmList[11][2])
                    elbow = (lmList[13][2])
                    hand = (lmList[15][2])
                    hip = (lmList[23][2])
                    foot = (lmList[27][2])
                
                # Left Arm
                else:
                    elbow_angle=detector.findAngle(img, 11, 13, 15)
                    if (elbow_angle > 180):
                        elbow_angle  = 360 - elbow_angle
                    hip_angle= detector.findAngle(img,11,23,25)
                    knee_angle= detector.findAngle(img,23,25,27)

                    # 그리기
                    detector.drawPoint(img, 11, 13, 15, 'elbow', elbow_angle)
                    detector.drawPoint(img, 11, 23, 25, 'hip', hip_angle)
                    detector.drawPoint(img, 23, 25, 27, 'knee', knee_angle)

                    # 각도를 퍼센트로 나타내는 코드
                    per = np.interp(elbow_angle, (80, 160), (100, 0))
                    bar = np.interp(elbow_angle, (80, 160), (650, 100))  # 앞에가 최소 뒤에가 최대
                    head = (lmList[0][2])
                    shoulder = (lmList[12][2])
                    elbow = (lmList[14][2])
                    hand = (lmList[16][2])
                    hip = (lmList[24][2])
                    foot = (lmList[28][2])

                keypoint = [head, shoulder, elbow, hand, hip, foot, int(elbow_angle), int(hip_angle),int(knee_angle)]  #CSV생성용 키포인트 데이터 생성
                
                cur_label = int(class_var.keypoint_pred(keypoint))
                keypoint_list.append(keypoint) 

                # 사전에 입력한 시작점과 끝점 외의 준비자세는 레이블을 0으로 둠
                answer = defineLabel(int(elbow_angle), int(hip_angle), int(knee_angle), int(cap.get(cv2.CAP_PROP_POS_FRAMES)), int(start_sec), int(end_sec))
                label_list.append([index, answer]) # index별로 뽑기위해 keypoint 리스트에 추가  
                
                # 카운트 확인
                count, isCorrect, isStart = pushup_instance.cal_count(cur_label)
                if(count == dataset.goal_cnt):
                    break

                # 정확도 체크 시작
                if isStart == True:
                    if dataset.accuracy == False and cur_label == 3:
                        dataset.accuracy = True
                    
                    if dataset.accuracy == True:
                        dataset.full_frames += 1
                        dataset.cur_light = 'green'

                    if(not(isCorrect) and dataset.accuracy == True):
                        dataset.incorrect_frames += 1
                        dataset.cur_light = 'red'

                add_Lights(img, dataset.cur_light)


            # 카운팅 횟수/게이지 바 그리기 
            #draw angle bar
            if(per == 100):
                img = cv2.ellipse(img, (1100,600), (90,90), 270, 0, per*3.6, (150, 250, 0), 15, 2)
            elif(per != 0):
                img = cv2.ellipse(img, (1100,600), (90,90), 270, 0, per*3.6, (255, 190, 0), 15, 2)
            
            #draw full-count bar
            if(int(count) != 0):
                img = cv2.ellipse(img, (1100,600), (105,105), 270, 0, int(count)*int(360/dataset.goal_cnt), (90, 90, 255), 10, 2) 
            elif(int(count) >= 10):
                img = cv2.ellipse(img, (1100,600), (105,105), 270, 0, int(count)*int(360/dataset.goal_cnt), (30, 30, 255), 10, 2)
                
            #draw curl count
            if(int(count) < 10):
                cv2.putText(img, str(int(count)), (1053,650), cv2.FONT_HERSHEY_PLAIN, 10, (180, 50, 50), 15)
            else:
                cv2.putText(img, str(int(count)), (1020,640), cv2.FONT_HERSHEY_PLAIN, 8, (180, 50, 50), 15)

            # # 프레임 그리기
            # cTime = time.time()
            # fps = 1/(cTime-pTime)
            # pTime = cTime
            # cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

            # 픽토그램 그리기
            img = add_Pictogram(img, cur_label)

            # 최종 출력
            cv2.imshow("Image",img)
            cv2.waitKey(1)

            # 동영상저장
            out.write(img)

        out.release()
        cv2.destroyAllWindows()


if __name__=="__main__":
    f = open("video_name.txt", 'r')

    video_name = f.readline().rstrip()
    while video_name: # 여러동영상에 대해 학습데이터 생성
        print(video_name)
        #home_trainer = HomeTrainer()
        dataset = GUI_data()
        HomeTrainer.run_pose_estimation(video_name, dataset)
        video_name = f.readline().rstrip()
