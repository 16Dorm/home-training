
def defineLabel(elbow_angle, hip_angle, knee_angle, frame, start_sec, end_sec):
    """
    keypoint데이터, min, max를 활용하여 레이블 0,1,2,3로 구분해줍니다.
    :return answer: 0,1,2
    
    // 준비자세 - 0
    사전에 입력한 start_sec과 end_sec을 기반으로 준비 자세를 레이블 0으로 구분

    // 아래 - 1
    data: 어깨-411 손-471
    sholder - hand 간 거리 : 60

    // 중간 - 2
    data: 어깨-348 손-480 
    sholder - hand 간 거리 : 132
    
    // 위 - 3
    data: 어깨-265 손-488 
    sholder - hand 간 거리 : 223
    
    answer=None
    sholder=0
    hand=1
    sholder_hand_length = keypoint[hand] - keypoint[sholder]
    print(sholder_hand_length)
    if(sholder_hand_length < 90) :
        answer=0
    elif ((sholder_hand_length >= 90) and (sholder_hand_length <180)) :
        answer=1
    elif ((sholder_hand_length >=170) and (sholder_hand_length < 240)) :
        answer=2


    세진
    엉덩이 165-195
    무릎 155-185

    요한
    엉덩이 150-210
    무릎 145-175

    병국
    엉덩이 155-185
    무릎 185-215

    광우
    엉덩이 160-190
    무릎 175-205

    유튜브
    엉덩이 150-180
    무릎 180-210
    """
    
    if (elbow_angle > 180):
        elbow_angle  = 360 - elbow_angle 

    answer = None
    #print(frame, start_sec*30, end_sec*30)
    if( frame < start_sec * 30 or frame > end_sec * 30):
        answer = 0
        print('-')
    else:
        if((hip_angle >= 150 and hip_angle <= 210) and (knee_angle >= 145 and knee_angle <=215)):
            if (elbow_angle  > 40 and elbow_angle  <90):
                answer=1
                # print(1)
            elif (elbow_angle  >=90 and elbow_angle  <140):
                answer=2
                # print(2)
            elif(elbow_angle >=140 and elbow_angle  < 180):
                answer=3
                # print(3)
        else:
            answer=0
            print(0)
    return answer