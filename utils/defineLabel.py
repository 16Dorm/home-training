
def defineLabel(keypoint):
    """
    keypoint데이터, min, max를 활용하여 레이블 0,1,2로 구분해줍니다.
    :return answer: 0,1,2
    
    //아래 - 0
    data: 어깨-411 손-471
    sholder - hand 간 거리 : 60

    // 중간 - 1
    data: 어깨-348 손-480 
    sholder - hand 간 거리 : 132
    
    // 위 - 2
    data: 어깨-265 손-488 
    sholder - hand 간 거리 : 223
    """
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
    
    return answer