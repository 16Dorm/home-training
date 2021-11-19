
# 3 2 1 2 3 순서가 되면 count + 1
# 단, 다음 순서 label 값이 들어오기 전에, label 0이 10회 이상 들어오면 리셋
# 단, 다음 순서 label 값이 들어오기 전에, 다른 label값이 3회 이상 들어오면 리셋

class Pushup_Counting:

    def __init__(self):
        self.label_table = [3,2,1,2]
        self.count = 0
        self.semi_count = 0
        self.zeors_count = 0
        self.deadline_count = 0
        self.pre_label = -1
        self.cur_label = -1
        self.prediction = 3

        self.result = False
        self.flag = -1
        self.start = False

    def cal_count(self, input_label):
        
        self.cur_label = input_label
        
        self.result = True

        '''
        if self.cur_label == 0:
            self.zeors_count += 1
            self.result = False
            if self.zeors_count > 10:
                self.semi_count = 0
                self.prediction = 3
                self.pre_label = self.cur_label
        elif self.pre_label != self.cur_label:
            if self.cur_label == self.prediction:
                if self.semi_count == 4:
                    self.count += 1
                    self.semi_count = 0
                self.zeors_count = 0
                self.semi_count += 1
                self.deadline_count = 0
                self.prediction = self.label_table[self.semi_count%4]
                self.pre_label = self.cur_label
            else:
                self.deadline_count += 1
                if self.deadline_count > 3:
                    self.result = False
                    self.semi_count = 0
                    self.zeors_count = 0
                    self.deadline_count = 0
                    self.prediction = 3
                    self.pre_label = -1
        
        if self.cur_label == 0:
            self.result=False
            self.zeors_count+=1

            if self.zeors_count > 3 and self.prediction != 3:
                self.semi_count = 0
                self.prediction = 3
                self.pre_label = self.cur_label


        elif self.cur_label == self.prediction:
            self.prediction += self.flag
            
            if self.prediction==1:
                self.flag *= -1
        
        '''
        #0이 들어올 시
        if self.cur_label == 0 :
            #아직 시작 전일 때
            if self.start == False :
                #시작하는 자세 3을 예측값으로
                self.prediction = 3
            
            #시작 후 일때
            else :
                #zero count 추가
                self.zeros_count +=1
                #3번이상 반복시 틀림.
                if self.zeros_count > 3:
                    self.result=False
  
        # 3이 맨처음 들어왔을 때 시작 해줌
        if self.cur_label == 3 and self.start == False :
            self.start = True
            self.pre_label = self.cur_label
            self.prediction += self.flag

        #이전값하고 똑같을 시
        if self.cur_label == self.pre_label and self.start == True:
            self.result=True
            self.zeros_count = 0


        # 예상값이 잘 들어 왔을 때
        if self.cur_label == self.prediction and self.start == True:
            self.result = True
            self.pre_label = self.cur_label
            self.zeros_count = 0

            # 예측값 다음 단계로
            self.prediction += self.flag
            
            # 예측값이 1일때는 다음 예측값은 + 를 해주어야 하므로 flag(-1)에 -1곱하기
            if self.prediction == 1 :
                self.flag *= -1
            
            # 한 사이클 다 돌았을 때
            if self.prediction == 3 and self.flag == 1:
                self.count+=1
                self.flag = -1
                self.prediction = 2
                self.pre_label = self.cur_label
        
        return self.count, self.result
