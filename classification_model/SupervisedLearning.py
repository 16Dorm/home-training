import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


class classification():
    # def __init__(self):
#데이터 전처리 및 가공
    
    def train_csv(self):
        global model
        df_train=pd.read_csv("./train/test.csv", names=["head","shoulder","elbow","hand","hip","foot","elbow_angle","hip_angle","knee_angle","path","label"]) #학습 데이터들
        df_test=pd.read_csv("./train/train.csv",names=["head","shoulder","elbow","hand","hip","foot","elbow_angle","hip_angle","knee_angle","path","label"]) #실시간 데이터

        #train 데이터
        x=df_train.drop(columns=['path','label'])
        y=df_train['label']

        #테스트 데이터
        x_test=df_test.drop(columns=['path','label'])
        y_test=df_test['label']

        # #데이터 전처리(분할)
        x_train, x_valid, y_train, y_valid = train_test_split(x,y, test_size=0.3,random_state=2020)

        # one_line_data=self.keypoint
        # print(x_train.shape)
        # print(x_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)

        # #학습 
        model=DecisionTreeClassifier()
        model.fit(x_train,y_train)

        # #평가
        # y_train_pred=model.predict(x_train)
        # y_test_pred=model.predict(x_test)
        # y_valid_pred=model.predict(x_valid)
        # y_test_pred=model.predict(x_test)
    def keypoint_pred(self,keypoint):
        one_line_pred=model.predict([keypoint])
        print(one_line_pred)

        # print(classification_report(y_train,y_train_pred))
        # print(classification_report(y_valid,y_valid_pred))
        # # print(y_test_pred)
        # print(classification_report(y_test,y_test_pred))
        # print(classification_report(y_train,y_test_pred))

if __name__ == "__main__":
    key_point_data=[292,269,377,487,317,450,164,161,197]
    class_var=classification()
    class_var.train_csv()
    class_var.keypoint_pred(key_point_data)
