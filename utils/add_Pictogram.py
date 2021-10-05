import cv2

def add_Pictogram(img, label):

    src1 = img
    src2 = cv2.imread("./utils/Pictogram_" + str(label+1) + ".png") #로고파일 읽기
    
    x_pos = 1030
    y_pos = 570

    rows, cols, channels = src2.shape
    roi = src1[y_pos:rows+y_pos,x_pos:cols+x_pos]

    src1_bg = cv2.bitwise_and(roi,roi)
    
    src2_fg = cv2.bitwise_and(src2,src2)
    
    dst = cv2.bitwise_or(src1_bg, src2_fg)
    
    src1[y_pos:rows+y_pos,x_pos:cols+x_pos] = dst
    
    return src1