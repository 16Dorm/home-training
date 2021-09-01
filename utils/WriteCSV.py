import os
import csv
import numpy as np
import pandas as pd

class WriteCSV():
    def __init__(self, path, datas, keypoints, video_name, ):
        """
        인스턴스 생성과 동시에 train.csv를 생성합니다.
        :param save_path: train.csv가 저장되어 있은 경로
        :param datas: [이미지idx, 정답레이블(0,1,2)]로 구성된 리스트
        :param keypoints: 키포인트 리스트
        :param video_name: 비디오 이름
        """
        self.path = path
        self.save_path = os.path.join(path, "train.csv")
        self.datas = datas
        self.keypoints = keypoints
        self.video_name = video_name
        self.coumns = ["head", "shoulder", "elbow", "hand", "hip", "foot", "angle", "image_path", "label"]
        
        if not os.path.exists(self.save_path):
            self._make_train_csv()

    def _make_train_csv(self):
        """
        train.csv를 생성합니다.
        """
        with open(self.save_path, 'w', newline='') as f:
            wr = csv.writer(f)
        f.close()

    def _write_train_csv(self, datas):
        """
        label 데이터를 train.csv에 한줄씩 추가해줍니다.
        """
        with open(self.save_path, 'w', newline='') as f:
            wr = csv.writer(f)
            for data in datas:
                #image_name = self.video_name+ "_image" + str(data[0])
                wr.writerow(data)
        f.close()
    
    def merge_train_csv(self):
        """
        train.csv에 이미 데이터가 있는경우 키포인트, 이미지경로, label데이터를 병합해줍니다.
        """ 
        origin_data = pd.read_csv(self.save_path, names=self.coumns)

        new_list = []
        for i in range(len(self.datas)):
            img_name = self.video_name + "_image" + str(self.datas[i][0]) + ".jpg"
            new = self.keypoints[i]
            new.append(str(self.path + 'image/' + img_name))
            new.append(int(self.datas[i][1]))
            new_list.append(new)
            print(new)
        new_data = pd.DataFrame(new_list, columns=self.coumns)

        merge_data = pd.concat([origin_data, new_data], axis=0) # 합치기
        merge_data['label'] = merge_data['label'].astype(int) # 정수
        merge_data = merge_data.drop_duplicates(['image_path'], keep='last').values # 덮어쓰기
        self._write_train_csv(merge_data)

if __name__ == "__main__":
    label_datas = [[0,0.0],[1,1.0],[2,0],[3,1],[4,1],[5,2.0]]
    keypoint_datas = [[470, 369, 515, 621, 124, 579, 0], [33, 369, 5, 621, 124, 579, 0], [1, 359, 5, 621, 124, 579, 0], [33, 3, 5, 621, 124, 579, 0], [8, 6, 5, 621, 124, 579, 0], [2, 369, 5, 621, 124, 579, 0]]
    writecsv = WriteCSV('./', label_datas, keypoint_datas, "pushup_0")
    #writecsv._write_train_csv(label_datas)
    writecsv.merge_train_csv()