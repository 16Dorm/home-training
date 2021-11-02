import math
import cv2
import numpy as np
import PoseModule as pm
from utils.defineLabel import defineLabel
from utils.WriteCSV import WriteCSV

def run_pose_estimation(video_name):
    cap = cv2.VideoCapture("./Video/" + video_name + ".mp4")

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

    # 스켈레톤 검출 및 라벨링 확인을 위한 변수
    index = 0
    label_list = []
    keypoint_list = []

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
                head = (lmList[0][2])
                shoulder = (lmList[12][2])
                elbow = (lmList[14][2])
                hand = (lmList[16][2])
                hip = (lmList[24][2])
                foot = (lmList[28][2])

            keypoint = [head, shoulder, elbow, hand, hip, foot, int(elbow_angle), int(hip_angle),int(knee_angle)]  #CSV생성용 키포인트 데이터 생성
            keypoint_list.append(keypoint) 

            # 사전에 입력한 시작점과 끝점 외의 준비자세는 레이블을 0으로 둠
            answer = defineLabel(int(elbow_angle), int(hip_angle), int(knee_angle), int(cap.get(cv2.CAP_PROP_POS_FRAMES)), int(start_sec), int(end_sec))
            label_list.append([index, answer]) # index별로 뽑기위해 keypoint 리스트에 추가  
            print(answer)
    writecsv = WriteCSV('./dataset/train/', "test.csv", label_list, keypoint_list, video_name)
    writecsv.merge_train_csv()

if __name__=="__main__":
    f = open("video_name.txt", 'r')

    video_name = f.readline().rstrip()
    while video_name: # 여러동영상에 대해 학습데이터 생성
        print(video_name)
        run_pose_estimation(video_name)
        video_name = f.readline().rstrip()
