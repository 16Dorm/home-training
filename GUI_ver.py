# GUI
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from time import sleep

# AI
from numpy.core import fromnumeric
from classification_model.SupervisedLearning import classification
from math import inf
import math
import cv2
import numpy as np
import time
import PoseModule as pm
from utils.add_Pictogram import add_Pictogram
from utils.defineLabel import defineLabel
from utils.WriteCSV import WriteCSV
from utils.Pushup_Counting import Pushup_Counting
from GUI.make_graph import make_graph

# GUI_1 -> AI -> GUI_2

class GUI_1(QWidget):
    def __init__(self):
        super().__init__()
        self.weight = 60
        self.goal_count = 3
        self.goal_set = 1
        self.isPressedConfirm = False
        self.setUI()

    def setUI(self):
        # 타이틀
        self.setWindowTitle('AI_Trainer')
        self.setWindowIcon(QIcon('./GUI/symbol_icon.png'))

        # 창 사이즈 고정
        # self.resize(1280, 720)
        self.setFixedSize(320, 250)
        #self.setFixedSize(1280, 720)

        # 레이아웃
        self.myLayout = QGridLayout()
        self.setLayout(self.myLayout)

        # 입력 제한자
        self.onlyInt = QIntValidator()

        # GIF
        label_gif = QLabel()
        self.movie = QMovie('./GUI/Pictogram_gif.gif', QByteArray(), self)
        self.movie.setCacheMode(QMovie.CacheAll)
        label_gif.setMovie(self.movie)
        self.movie.start()
        self.myLayout.addWidget(label_gif, 0,0, 1,3)

        # 몸무게 라벨
        label_weight = QLabel('몸무게를 입력해 주세요.', self)
        self.myLayout.addWidget(label_weight, 2,0)

        # 몸무게 텍스트박스
        lineedit1 = QLineEdit(self)
        lineedit1.setText(str(self.weight))
        lineedit1.setValidator(self.onlyInt)
        self.myLayout.addWidget(lineedit1, 2,1)

        # 횟수 라벨
        label2 = QLabel('목표 횟수를 입력해 주세요.', self)
        self.myLayout.addWidget(label2, 3,0)

        # 횟수 콤보박스
        combo1 = QComboBox(self)
        combo1.addItem('3')
        combo1.addItem('5')
        combo1.addItem('10')
        combo1.addItem('12')
        combo1.addItem('20')
        combo1.addItem('100')
        combo1.activated[str].connect(lambda :self.selectedComboItem(combo1, "cnt"))
        self.myLayout.addWidget(combo1, 3,1)

        # 세트 라벨
        label3 = QLabel('목표 세트를 입력해 주세요.', self)
        self.myLayout.addWidget(label3, 4,0)

        # 세트 콤보박스
        combo2 = QComboBox(self)
        combo2.addItem('1')
        combo2.addItem('2')
        combo2.addItem('3')
        combo2.addItem('4')
        combo2.addItem('5')
        combo2.activated[str].connect(lambda :self.selectedComboItem(combo2, "set"))
        self.myLayout.addWidget(combo2, 4,1)

        # 확인 버튼
        button1 = QPushButton('확인', self)
        self.myLayout.addWidget(button1, 4,2)
        button1.clicked.connect(lambda: self.btnClickedEvent(lineedit1))

    def selectedComboItem(self,text,type):
        if type == "cnt":
            self.goal_count = int(text.currentText())
            print("cnt : ", self.goal_count)
        elif type == "set":
            self.goal_set = int(text.currentText())
            print("set : ", self.goal_set)
        else:
            print("combobox type check error")

    def btnClickedEvent(self, txt):
        self.weight = int(txt.text())
        self.isPressedConfirm = True
        QCoreApplication.instance().quit()

class GUI_2(QWidget):
    def __init__(self):
        super().__init__()
        self.setUI()

    def setUI(self):
        # 타이틀
        self.setWindowTitle('AI_Trainer')
        self.setWindowIcon(QIcon('./GUI/symbol_icon.png'))

        # 창 사이즈 고정
        self.setFixedSize(1280, 720)

        # 레이아웃
        self.myLayout = QGridLayout()
        self.setLayout(self.myLayout)

        # graph 생성
        make_graph(AI_intance.incorrect_cnt, AI_intance.full_frames)

        # 이미지 라벨
        label1 = QLabel(self)
        pixmap = QPixmap('./GUI/graph.png')
        pixmap =pixmap.scaled(int(pixmap.width()),int(pixmap.height()))
        label1.setPixmap(pixmap)
        self.myLayout.addWidget(label1, 0,0, 2,2)

        # 이미지 라벨
        label3 = QLabel(self)

        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:,:] = [240, 240, 240]
        img = cv2.ellipse(img, (150,150), (110,110), 270, 0, 360, (255, 255, 255), 20, 10)
        height, width, channel = img.shape
        bytesPerLine = 3 * width

        img = cv2.ellipse(img, (150,150), (110,110), 270, 0, ((1-(AI_intance.incorrect_cnt/AI_intance.full_frames)) * 100)*3.6, (150, 250, 0), 20, 10)
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        pixmap =pixmap.scaled(int(pixmap.width()),int(pixmap.height()))
        
        label3.setPixmap(pixmap)
        self.myLayout.addWidget(label3, 2,0, 2,2)

        # 이미지 라벨
        label33 = QLabel(self)

        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:,:] = [240, 240, 240]
        img = cv2.ellipse(img, (150,150), (110,110), 270, 0, 360, (255, 0, 0), 18, 10)
        height, width, channel = img.shape
        bytesPerLine = 3 * width

        img = cv2.ellipse(img, (150,150), (110,110), 270, 0, ((1-(AI_intance.incorrect_cnt/AI_intance.full_frames)) * 100)*3.6, (150, 250, 0), 20, 10)
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        pixmap =pixmap.scaled(int(pixmap.width()),int(pixmap.height()))
        
        label33.setPixmap(pixmap)
        self.myLayout.addWidget(label33, 2,3, 2,4)

        # 이미지 라벨
        label333 = QLabel(self)

        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:,:] = [240, 240, 240]
        img = cv2.ellipse(img, (150,150), (110,110), 270, 0, 360, (240, 240, 240), 20, 10)
        height, width, channel = img.shape
        bytesPerLine = 3 * width

        img = cv2.ellipse(img, (150,150), (110,110), 270, 0, ((1-(AI_intance.incorrect_cnt/AI_intance.full_frames)) * 100)*3.6, (150, 250, 0), 20, 10)
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        pixmap =pixmap.scaled(int(pixmap.width()),int(pixmap.height()))
        
        label333.setPixmap(pixmap)
        self.myLayout.addWidget(label333, 0,3, 2,4)

        # 횟수 라벨
        label4 = QLabel('결과 이미지 및 등등', self)
        self.myLayout.addWidget(label4, 2,5)

        self.show()
  
class AI_Train():

    def __init__(self):
        self.accuracy = False
        self.full_frames = 0
        self.incorrect_cnt = 0

    def run_pose_estimation(self, video_name, goal_cnt, goal_set):
        
        print(goal_cnt, goal_set)
        
        cap = cv2.VideoCapture("./Video/" + video_name + ".mp4")
        #cap=cv2.VideoCapture(0) #카메라 번호
        
        # 사전 준비시간을 label0으로 잘라내기 위한 작업
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
        detector =pm.poseDetector(video_name)

        while True:
            success, img =cap.read()
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
                count, isCorrect = pushup_instance.cal_count(cur_label)

                # 정확도 체크 시작
                if self.accuracy == False and cur_label == 3:
                    self.accuracy = True
                
                if self.accuracy == True:
                    self.full_frames += 1

                if(not(isCorrect) and self.accuracy == True):
                    self.incorrect_cnt += 1

            # 카운팅 횟수/게이지 바 그리기 
            #draw angle bar
            if(per == 100):
                img = cv2.ellipse(img, (1100,600), (90,90), 270, 0, per*3.6, (150, 250, 0), 15, 2)
            elif(per != 0):
                img = cv2.ellipse(img, (1100,600), (90,90), 270, 0, per*3.6, (255, 190, 0), 15, 2)
            
            #draw full-count bar
            if(int(count) != 0):
                img = cv2.ellipse(img, (1100,600), (105,105), 270, 0, int(count)*int(360/goal_cnt), (90, 90, 255), 10, 2) 
            elif(int(count) >= 10):
                img = cv2.ellipse(img, (1100,600), (105,105), 270, 0, int(count)*int(360/goal_cnt), (30, 30, 255), 10, 2)
                
            #draw curl count
            if(int(count) < 10):
                cv2.putText(img, str(int(count)), (1053,650), cv2.FONT_HERSHEY_PLAIN, 10, (180, 50, 50), 15)
            else:
                cv2.putText(img, str(int(count)), (1020,640), cv2.FONT_HERSHEY_PLAIN, 8, (180, 50, 50), 15)

            # 프레임 그리기
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

            # 픽토그램 그리기
            img = add_Pictogram(img, int(per/34))

            # 최종 출력
            cv2.imshow("Image",img)
            cv2.waitKey(1) 

        #final_max=[max(head),max(shoulder),max(elbow),max(hand),max(hip),max(foot)]
        #final_min=[min(head),min(shoulder),min(elbow),min(hand),min(hip),min(foot)]
        #print(f"final_max: {final_max}")
        #print(f"final_min: {final_min}")
        
        # WriteCSV 제거
        #writecsv = WriteCSV('./dataset/train/', "train.csv", label_list, keypoint_list, video_name)
        #writecsv.merge_train_csv()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    app = QApplication(sys.argv)

    # 1차 GUI
    mywindow_1 = GUI_1()
    mywindow_1.show()
    app.exec_()
    mywindow_1.close()

    if mywindow_1.isPressedConfirm == True:
        # AI
        # --code--
        AI_intance = AI_Train()
        AI_intance.run_pose_estimation("pushup_08", mywindow_1.goal_count, mywindow_1.goal_set)
        
        # 2차 GUI
        mywindow_2 = GUI_2()
        mywindow_2.show()
        sys.exit(app.exec_())

    '''
    추후 할 것
    0. 몸무게 입력 
    1. 결과 창 꾸미기 (정확도/칼로리?)
    2. 목표 횟수 도달 후 세트 변경시 어떻게 할지 생각하기
    3. 실시간으로 수행 할 때도 video_name이 있어아햐는 문제 해결하기 (detector =pm.poseDetector(video_name) <-- 이게 문제인듯?)
    '''