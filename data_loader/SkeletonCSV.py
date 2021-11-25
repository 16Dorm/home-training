import os
import pandas as pd

class SkeletonCSV():
    def __init__(self, data_dir, trsfm=None):
        self.columns = ["head", "shoulder", "elbow", "hand", "hip", "foot", "elbow_angle", "hip_angle", "knee_angle", "image_path", "label"]
        self.trsfm = trsfm

        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval')

        train_csv_path=os.path.join(self.train_dir, "train.csv")
        test_csv_path=os.path.join(self.train_dir, "test.csv")
        print(train_csv_path)
        self.df = pd.read_csv(train_csv_path, names=self.columns)
        self.test_df = pd.read_csv(test_csv_path, names=self.columns)



if __name__ == "__main__":
    dataloader = SkeletonCSV("./dataset")
    train_dataloader, vliad_dataloader = dataloader.split_validation()
    for data in train_dataloader:
        print(data)