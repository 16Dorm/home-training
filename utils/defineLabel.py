
def defineLabel(angle, frame, start_sec, end_sec):
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
    """
    
    answer = None
    #print(frame, start_sec*30, end_sec*30)
    if( frame < start_sec * 30 or frame > end_sec * 30):
        answer = 0
        print('-')
    else:
        if (angle > 40 and angle <90):
            answer=1
            print(1)
        elif (angle >=90 and angle <140):
            answer=2
            print(2)
        elif(angle>=140 and angle < 180):
            answer=3
            print(3)
        else:
            answer=0
            print(0)
    return answer