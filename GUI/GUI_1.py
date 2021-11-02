import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

goal_count = 0
goal_set = 0

class GUI_1(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setUI()

    def setUI(self):
        # 타이틀
        self.setWindowTitle('AI_Trainer')
        self.setWindowIcon(QIcon('./GUI/symbol_icon.png'))

        # 창 사이즈 고정
        # self.resize(300, 200)
        self.setFixedSize(315, 200)

        # 레이아웃
        self.myLayout = QGridLayout()
        self.setLayout(self.myLayout)

        # GIF
        label = QLabel()
        self.movie = QMovie('./GUI/Pictogram_gif.gif', QByteArray(), self)
        self.movie.setCacheMode(QMovie.CacheAll)
        label.setMovie(self.movie)
        self.movie.start()
        self.myLayout.addWidget(label, 0,0, 1,3)

        # 횟수 라벨
        label2 = QLabel('목표 횟수를 입력해 주세요.', self)
        self.myLayout.addWidget(label2, 1,0)

        # 횟수 콤보박스
        combo1 = QComboBox(self)
        combo1.addItem('3')
        combo1.addItem('5')
        combo1.addItem('10')
        combo1.addItem('12')
        combo1.addItem('20')
        combo1.addItem('100')
        combo1.activated[str].connect(lambda :self.selectedComboItem(combo1, "cnt"))
        self.myLayout.addWidget(combo1, 1,1)

        # 세트 라벨
        label3 = QLabel('목표 세트를 입력해 주세요.', self)
        self.myLayout.addWidget(label3, 2,0)

        # 세트 콤보박스
        combo2 = QComboBox(self)
        combo2.addItem('1')
        combo2.addItem('2')
        combo2.addItem('3')
        combo2.addItem('4')
        combo2.addItem('5')
        combo2.activated[str].connect(lambda :self.selectedComboItem(combo2, "set"))
        self.myLayout.addWidget(combo2, 2,1)

        # 확인 버튼
        button1 = QPushButton('확인', self)
        button1.clicked.connect(QCoreApplication.instance().quit)
        self.myLayout.addWidget(button1, 2,2)

    def selectedComboItem(self,text,type):
        if type == "cnt":
            goal_count = int(text.currentText())
            print("cnt : ", goal_count)
        elif type == "set":
            goal_set = int(text.currentText())
            print("set : ", goal_set)
        else:
            print("combobox type check error")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 1차 GUI
    mywindow = GUI_1()
    mywindow.show()
    sys.exit(app.exec_())