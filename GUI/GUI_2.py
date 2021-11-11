import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np
import cv2
from time import sleep

from make_graph import make_graph

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

        # graph 생성
        #make_graph(AI_intance.incorrect_cnt, AI_intance.full_frmes)
        make_graph(12,100)

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

        loop_cnt = 1
        while 1:
            if loop_cnt > 100:
                break
            img = cv2.ellipse(img, (150,150), (110,110), 270, 0, loop_cnt*3.6, (150, 250, 0), 20, 10)
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap(qImg)
            pixmap =pixmap.scaled(int(pixmap.width()),int(pixmap.height()))
            
            label3.setPixmap(pixmap)
            self.myLayout.addWidget(label3, 2,0, 2,2)

            label3.repaint()
            self.show()
            sleep(0.01)
            loop_cnt += 1

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