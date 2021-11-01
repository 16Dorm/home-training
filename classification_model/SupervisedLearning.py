import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_loader.SkeletonCSV import SkeletonCSV

class classification():
    def __init__(self):
        dataset_dir = './dataset'
        skeleton_csv = SkeletonCSV(dataset_dir)
        self.train, self.valid = train_test_split(skeleton_csv.df, test_size=0.2, random_state=42, stratify=skeleton_csv.df.to_numpy()[:,-1])
        self.test = skeleton_csv.test_df
        # 13148, 3287 data length
        
        #df_train=pd.read_csv("./dataset/train/train.csv", names=["head","shoulder","elbow","hand","hip","foot","elbow_angle","hip_angle","knee_angle","path","label"]) #학습 데이터들
        #df_test=pd.read_csv("./dataset/train/test.csv",names=["head","shoulder","elbow","hand","hip","foot","elbow_angle","hip_angle","knee_angle","path","label"]) #실시간 데이터

        # #train 데이터
        self.x_train=self.train.drop(columns=['image_path','label'])
        self.y_train=self.train['label']

        # #valid 데이터
        self.x_valid=self.valid.drop(columns=['image_path','label'])
        self.y_valid=self.valid['label']

        # #test 데이터
        self.x_test=self.test.drop(columns=['image_path','label'])
        self.y_test=self.test['label']
        
        self.model=DecisionTreeClassifier()
        self.y_train_pred = None
        self.y_valid_pred = None
        self.y_test_pred = None


    def train_csv(self):
        # one_line_data=self.keypoint
        # print(x_train.shape)
        # print(x_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)

        # 학습 
        self.model.fit(self.x_train,self.y_train)

        # 평가
        self.y_train_pred=self.model.predict(self.x_train)
        self.y_valid_pred=self.model.predict(self.x_valid)
        self.y_test_pred=self.model.predict(self.x_test)
        

    def keypoint_pred(self,keypoint):
        one_line_pred=self.model.predict([keypoint])
        print(one_line_pred)
        print('--------- train result ------------')
        print(classification_report(self.y_train,self.y_train_pred))
        print('--------- valid result ------------')
        print(classification_report(self.y_valid,self.y_valid_pred))
        print('--------- test result ------------')
        print(classification_report(self.y_test,self.y_test_pred))

if __name__ == "__main__":
    key_point_data=[292,269,377,487,317,450,164,161,197]
    class_var=classification()
    class_var.train_csv()
    class_var.keypoint_pred(key_point_data)