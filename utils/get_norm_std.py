import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tqdm import tqdm
from time import time
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data_loader.SkeletonDataset import SkeletonDataset
from data_loader.SkeletonDataLoader import SkeletonDataLoader


# 1. 데이터 경로 및 하이퍼파라미터
dataset_dir = './dataset'
num_workers = 0
batch_size = 8

A_transforms = A.Compose([
                    ToTensorV2()
                ])


# 2. 학습데이터의 정규분포값 구하기
skeleton_dataloader = SkeletonDataLoader(dataset_dir, batch_size=batch_size, trsfm=A_transforms, num_worker=num_workers)
dataset, dataloader = skeleton_dataloader.get_dataset_dataloader()
print(len(dataloader))

N_CHANNELS =3

before = time()
mean = torch.zeros(N_CHANNELS)
std = torch.zeros(N_CHANNELS)

print('==> Computing mean and std..')
for inputs, _labels in tqdm(dataloader):
    for i in range(N_CHANNELS):
        mean[i] += (inputs[:,i,:,:] *1.0).mean()
        std[i] += (inputs[:,i,:,:]* 1.0).std()
mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)
print("time elapsed: ", time()-before)

# 16435 answer
# tensor([1.0462, 2.4636, 0.4279]) 
# tensor([12.2819, 22.4455,  3.6525])