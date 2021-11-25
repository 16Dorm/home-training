import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from data_loader.SkeletonCSV import SkeletonCSV

class classification():
    def __init__(self):
        # SkeletonCSV를 활용 df를 불러오기
        dataset_dir = './dataset'
        skeleton_csv = SkeletonCSV(dataset_dir)
        self.train, self.valid = train_test_split(skeleton_csv.df, test_size=0.2, random_state=42, stratify=skeleton_csv.df.to_numpy()[:,-1])
        self.test = skeleton_csv.test_df
        
        # train 데이터
        self.x_train=self.train.drop(columns=['image_path','label']).values
        self.y_train=self.train['label'].values
        # valid 데이터
        self.x_valid=self.valid.drop(columns=['image_path','label']).values
        self.y_valid=self.valid['label'].values
        # test 데이터
        self.x_test=self.test.drop(columns=['image_path','label']).values
        self.y_test=self.test['label'].values
        
        self.model=DecisionTreeClassifier()
        self.y_train_pred = None
        self.y_valid_pred = None
        self.y_test_pred = None


    def train_csv(self):
        """ 모델 학습 및 성능 평가 함수 """
        self.model.fit(self.x_train,self.y_train) # 학습
        self.y_train_pred=self.model.predict(self.x_train) # 성능평가
        self.y_valid_pred=self.model.predict(self.x_valid)
        self.y_test_pred=self.model.predict(self.x_test)
        

    def keypoint_pred(self,keypoint):
        """ 실시간 키포인트 예측 함수"""
        one_line_pred=self.model.predict([keypoint])
        return one_line_pred


if __name__ == "__main__":
    class_var=classification()
    class_var.train_csv()

    print('--------- data shape ------------')
    print(class_var.x_train.shape)
    print(class_var.x_test.shape)
    print(class_var.y_train.shape)
    print(class_var.y_test.shape)

    print('--------- train result ------------')
    print(classification_report(class_var.y_train,class_var.y_train_pred))
    print('--------- valid result ------------')
    print(classification_report(class_var.y_valid,class_var.y_valid_pred))
    print('--------- test result ------------')
    print(classification_report(class_var.y_test,class_var.y_test_pred))