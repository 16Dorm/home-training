import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class GUI_2(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setUI()

    def setUI(self):
        # 타이틀
        self.setWindowTitle('AI_Trainer')
        self.setWindowIcon(QIcon('./GUI/symbol_icon.png'))

        # 창 사이즈 고정
        self.setFixedSize(1200, 800)

        # 레이아웃
        self.myLayout = QGridLayout()
        self.setLayout(self.myLayout)

        # 이미지 라벨
        label1 = QLabel(self)
        pixmap = QPixmap('./GUI/graph.png')
        pixmap =pixmap.scaled(int(pixmap.width()),int(pixmap.height()))
        label1.setPixmap(pixmap)
        self.myLayout.addWidget(label1, 0,0, 2,2)

        # 설명 라벨
        label2 = QLabel('결과 이미지 및 등등', self)
        self.myLayout.addWidget(label2, 0,2)

        # 이미지 라벨
        label3 = QLabel(self)
        pixmap = QPixmap('./GUI/graph.png')
        pixmap =pixmap.scaled(int(pixmap.width()),int(pixmap.height()))
        label3.setPixmap(pixmap)
        self.myLayout.addWidget(label3, 2,0, 2,2)

        # 횟수 라벨
        label4 = QLabel('결과 이미지 및 등등', self)
        self.myLayout.addWidget(label4, 2,2)

        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 2차 GUI
    mywindow = GUI_2()
    mywindow.show()
    sys.exit(app.exec_())