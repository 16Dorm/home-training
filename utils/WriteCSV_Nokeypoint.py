import os
import csv


class WriteCSV():
    def __init__(self, path, datas):
        """
        인스턴스 생성과 동시에 train.csv를 생성합니다.
        :param save_path: train.csv가 저장되어 있은 경로
        :param datas: [이미지idx, 정답레이블(0,1,2)]로 구성된 리스트
        :param keypoints: 키포인트 리스트
        :param video_name: 비디오 이름
        """
        self.path = path
        self.save_path = os.path.join(path, "train_Nokeypoint.csv")
        self.datas = datas
        
    def _write_train_csv(self, datas):
        """
        label 데이터를 train.csv에 한줄씩 추가해줍니다.
        """
        with open(self.save_path, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerows(datas)
        f.close()


if __name__ == "__main__":
    f = open('./train/train.csv')

    with open('./train/train_Nokeypoint.csv', 'w', newline='') as f2:
            wr = csv.writer(f2)
    
    f2.close()

    rdr = csv.reader(f)
    for line in rdr:
        print([line[7:9]])
        writecsv = WriteCSV('./train',  line[7:9])
        writecsv._write_train_csv([line[7:9]])

    f.close()
    
    