import os
from .SkeletonDataset import SkeletonDataset
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class SkeletonDataLoader():
    def __init__(self, data_dir, batch_size, num_worker, shuffle=True, validation_split=0.2, trsfm=None):
        self.columns = ["head", "shoulder", "elbow", "hand", "hip", "foot", "elbow_angle", "hip_angle", "knee_angle", "image_path", "label"]

        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval')

        csv_path=os.path.join(self.train_dir, "train.csv")
        print(csv_path)
        self.df = pd.read_csv(csv_path, names=self.columns)

        self.data = SkeletonDataset(self.df, trsfm)
        self.train_dataset, self.valid_dataset = train_test_split(self.data, test_size=validation_split, random_state=42, stratify=self.df.to_numpy()[:,-1])

        self.train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=shuffle, num_workers=num_worker, pin_memory=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size, shuffle=shuffle, num_workers=num_worker, pin_memory=True)

        #assert trsfm is not None, "Set the transform on train set"

    def split_validation(self):
        return self.train_dataloader, self.valid_dataloader

if __name__ == "__main__":
    dataloader = SkeletonDataLoader("./", batch_size=16, validation_split=0.2, num_workers=8)
    train_dataloader, vliad_dataloader = dataloader.split_validation()
    for data in train_dataloader:
        print(data)