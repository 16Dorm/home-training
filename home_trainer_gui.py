# GUI
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from time import sleep
import os
import cv2

from GUI.make_graph import make_graph
from home_trainer import HomeTrainer
# GUI_form -> AI -> GUI_result

wantcam = False

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

class GUI_form(QWidget):
    def __init__(self, dataset):
        super().__init__()
        self.setUI(dataset)

    def setUI(self, dataset):
        # 타이틀
        self.setWindowTitle('HomeTrainer')
        self.setWindowIcon(QIcon('./GUI/symbol_icon.png'))

        # 창 사이즈 고정
        # self.resize(1280, 720)
        self.setFixedSize(320, 250)

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
        lineedit1.setText(str(dataset.weight))
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
        combo1.activated[str].connect(lambda :self.selectedComboItem(combo1, "cnt", dataset))
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
        combo2.activated[str].connect(lambda :self.selectedComboItem(combo2, "set", dataset))
        self.myLayout.addWidget(combo2, 4,1)

        # 몸무게 라벨
        label_interval = QLabel('쉬는 시간을 입력해 주세요.', self)
        self.myLayout.addWidget(label_interval, 5,0)

        # 세트 콤보박스
        # 몸무게 텍스트박스
        lineedit2 = QLineEdit(self)
        lineedit2.setText(str(dataset.interval_sec_per_set))
        lineedit2.setValidator(self.onlyInt)
        self.myLayout.addWidget(lineedit2, 5,1)


        # 확인 버튼
        button1 = QPushButton('확인', self)
        self.myLayout.addWidget(button1, 5,2)
        button1.clicked.connect(lambda: self.btnClickedEvent(lineedit1, lineedit2))
        

    def selectedComboItem(self,text,type, dataset):
        if type == "cnt":
            dataset.goal_cnt = int(text.currentText())
            print("cnt : ", dataset.goal_cnt)
        elif type == "set":
            dataset.goal_set = int(text.currentText())
            print("set : ", dataset.goal_set)
        else:
            print("combobox type check error")

    def btnClickedEvent(self, weight_txt, interval_txt):
        dataset.weight = int(weight_txt.text())
        dataset.interval_sec_per_set = int(interval_txt.text())
        dataset.isPressedConfirm = True
        QCoreApplication.instance().quit()

class GUI_timer(QWidget):
    def __init__(self, dataset):
        super().__init__()
        self.lcd = QLCDNumber()
        self.interval_time = dataset.interval_sec_per_set
        self.setUI(dataset)

    def setUI(self, dataset):
        # 타이틀
        self.setWindowTitle('HomeTrainer')
        self.setWindowIcon(QIcon('./GUI/symbol_icon.png'))

        # 창 사이즈 고정
        img = cv2.imread('./play_results/graph_' + str(dataset.cur_set_num) +'.png')
        h, w, c = img.shape
        self.setFixedSize(int(w)+int(w)/10, 600)

        # 레이아웃 설정
        self.mylayout = QVBoxLayout()

        # 정확도 그래프 출력
        label1 = QLabel(self)
        # pixmap = QPixmap('./GUI/graph_' + str(dataset.cur_set_num) + '.png')
        pixmap = QPixmap('./play_results/graph_' + str(dataset.cur_set_num) + '.png')
        pixmap =pixmap.scaled(int(pixmap.width()),int(pixmap.height()))
        label1.setPixmap(pixmap)
        self.mylayout.addWidget(label1)

        # 타이머 생성
        self.timer = QTimer(self)

        # 1000ms마다 timeout실행
        self.timer.setInterval(1000)
        self.timer.timeout.connect(lambda: self.timeout(dataset))

        # 글씨 칸 조절
        self.lcd.setDigitCount(3)

        # LCD에 숫자 띄우기 (숫자 맞추기 위해 -1 실행)
        self.lcd.display(self.interval_time)
        self.interval_time -= 1
        
        # 레이아웃에 따른 위치 설정
        self.mylayout.addWidget(self.lcd)
        self.setLayout(self.mylayout)

        # 타이머 시작
        self.timer.start()

    def timeout(self, dataset):
        currentTime = self.interval_time
        self.interval_time -= 1
        self.lcd.display(currentTime)
        
        # 타이머 종료
        if self.interval_time < 0:
            self.timer.stop()
            QCoreApplication.instance().quit()

class GUI_result(QWidget):
    def __init__(self, dataset):
        super().__init__()
        self.setUI(dataset)

    def setUI(self, dataset):
        # 타이틀
        self.setWindowTitle('HomeTrainer')
        self.setWindowIcon(QIcon('./GUI/symbol_icon.png'))

        # # 창 사이즈 고정
        img = cv2.imread('./play_results/graph_total.png')
        h, w, c = img.shape
        self.setFixedSize(int(w)+int(w)/10, 500)

        # 레이아웃
        # QVBoxLayout()
        self.myLayout = QVBoxLayout()
        self.myMiniLayout = QGridLayout()
        self.setLayout(self.myLayout)

        # graph 생성
        make_graph(dataset.incorrect_frames_total, dataset.full_frames_total, 'total')

        # 이미지 라벨
        label1 = QLabel(self)
        pixmap = QPixmap('./play_results/graph_total.png')
        pixmap =pixmap.scaled(int(pixmap.width()),int(pixmap.height()))
        label1.setPixmap(pixmap)
        self.myLayout.addWidget(label1)

        # 칼로리 계산 (100개 기준 몸무게 당 칼로리)
        if(dataset.weight <= 60):
            cal = 28000
        elif(dataset.weight <= 70):
            cal = 34000
        elif(dataset.weight <=80):
            cal = 41000
        elif(dataset.weight <=90):
            cal = 49000
        else:
            cal = 59000
        
        result_cnt = dataset.goal_cnt * dataset.goal_set
        cal = cal * (result_cnt /100)
        kcal = cal / 1000

        label4 = QLabel('사용자의 몸무게 ' + str(dataset.weight) + 'kg으로')
        label5 = QLabel('총 ' + str(result_cnt) + '번 팔굽혀펴기 결과,')
        label6 = QLabel('소모된 칼로리는 ' + str(kcal) + 'kcal입니다. ', self)
        self.myLayout.addWidget(label4)
        self.myLayout.addWidget(label5)
        self.myLayout.addWidget(label6)


        # 횟수 콤보박스
        combo1 = QComboBox(self)
        for i in range(dataset.goal_set):
            combo1.addItem(str(i+1))
        combo1.activated[str].connect(lambda :self.selectedComboItem(combo1, dataset))
        self.myMiniLayout.addWidget(combo1, 0,0)

        # 다시 시작 버튼
        btn_replay = QPushButton("RePlay")
        btn_replay.clicked.connect(self.Clicked_Replay_Button)
        self.myMiniLayout.addWidget(btn_replay, 0,1)

        # 홈 버튼
        btn_home = QPushButton("Home")
        btn_home.clicked.connect(self.Clicked_Home_Button)
        self.myMiniLayout.addWidget(btn_home, 0,2)
        
        # 종료 버튼
        btn_exit = QPushButton("Exit")
        btn_exit.clicked.connect(self.Clicked_Exit_Button)
        self.myMiniLayout.addWidget(btn_exit, 0,3)

        self.myLayout.addLayout(self.myMiniLayout)

        self.show()

    def selectedComboItem(self, text, dataset):
        # index 접근을 위한 -1
        dataset.want_replay_set = int(text.currentText())-1
        print(dataset.want_replay_set)

    def Clicked_Replay_Button(self):
        dataset.isReplay = True
        # dataset.want_replay_set를 통해서 리플레이 구현하기
        print('리플레이 구현하기')

        result_cap = cv2.VideoCapture('./play_results/output_' + str(dataset.want_replay_set) + '.avi')
        
        while result_cap.isOpened():
            run, frame = result_cap.read()
            if not run:
                break
            cv2.imshow('video', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        result_cap.release()
        cv2.destroyAllWindows()

    def Clicked_Home_Button(self):
        dataset.Home = True
        QCoreApplication.instance().quit()
        filePath='./play_results'
        if os.path.exists(filePath):
            for file in os.scandir(filePath):
                os.remove(file.path)
            print("Remove All file")
        else:
            print("Directory Not found")
    
    def Clicked_Exit_Button(self):
        dataset.Exit = True
        QCoreApplication.instance().quit()
        filePath='./play_results'
        if os.path.exists(filePath):
            for file in os.scandir(filePath):
                os.remove(file.path)
            print("Remove All file")
        else:
            print("Directory Not found")
        
if __name__ == '__main__':

    f = open("video_name.txt", 'r')
    video_name = f.readline().rstrip()

    app = QApplication(sys.argv)
    
    dataset = GUI_data()

    if not os.path.exists('play_results'):
        os.mkdir('play_results')


    while(dataset.Home):
        # 변수 초기화
        dataset.Home = False
        dataset.cur_set_num = 0 

        # 1차 GUI
        mywindow_1 = GUI_form(dataset)
        mywindow_1.show()
        app.exec_()
        mywindow_1.close()

        if dataset.isPressedConfirm == True:
            # 변수 초기화
            dataset.isPressedConfirm = False

            for i in range(dataset.goal_set):
                # AI
                HomeTrainer.run_pose_estimation(video_name, dataset)

                # graph 생성
                make_graph(dataset.incorrect_frames, dataset.full_frames, dataset.cur_set_num)
            
                # 마지막 세트 이후엔 쉬는시간 X
                if i < (dataset.goal_set-1):
                    # Timer
                    Timer = GUI_timer(dataset)
                    Timer.show()
                    app.exec_()
                    Timer.close()
                
                # 정확도 graph 업데이트/초기화
                dataset.full_frames_total += dataset.full_frames
                dataset.incorrect_frames_total += dataset.incorrect_frames
                dataset.full_frames = 0
                dataset.incorrect_frames = 0
                dataset.cur_light = 'black'

                # 현재 set 카운트
                dataset.cur_set_num += 1

            # 2차 GUI
            mywindow_2 = GUI_result(dataset)
            mywindow_2.show()
            app.exec_()
            mywindow_2.close()

    sys.exit()
    '''
    추후 할 것

    1. 1세트 끝나면 타이머 이용하기
        1) 1세트가 끝나면 영상 종료 후 타이머 GUI 생성 + 정확도 표기? -> 해결
        2) 타이머가 종료되면 다시 영상 시작 -> 해결
        3) 모든 세트 끝나면 결과창 출력 -> 해결

    2. 결과 창 꾸미기
        1) 칼로리 출력 -> 해결
        2) 정확도 출력 -> 해결
        3) 다시보기 -> 해결

    3. 다시보기 버튼 실행 시 영상 보여주기 -> 해결(2번)
        1)타이머로 끊으면 모든 세트에 해당하는 영상을 봐야하나?????
        2)콤보박스로 만들어서 원하는 세트 선택해서 볼 수 있게 하는게 더 나을듯

    4. 실시간으로 수행 할 때도 video_name이 있어아햐는 문제 해결하기 (detector =pm.poseDetector(video_name) <-- 이게 문제인듯?)
        -> 광우가 해결했나??
    '''