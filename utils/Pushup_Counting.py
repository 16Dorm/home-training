
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

    def cal_count(self, input_label):
        
        self.cur_label = input_label
        
        self.result = True

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

        return self.count, self.result
