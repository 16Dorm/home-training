import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from albumentations.augmentations.transforms import HorizontalFlip
from albumentations.core.serialization import save
import numpy as np
import pandas as pd
import torch

from data_loader.SkeletonDataset import SkeletonDataset
from data_loader.SkeletonCSV import SkeletonCSV
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import torchmetrics
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import copy
import random
import wandb
wandb.login()


# 1. 데이터 경로 및 하이퍼파라미터
dataset_dir = './dataset'
save_dir = './results/'
MODEL_NAME = "efficientnet_b0"
num_workers = 0
learning_rate = 1e-4
batch_size = 32
step_size = 5
epochs = 60
early_stop =5
seed = 2021

A_transforms = {
    'train' : A.Compose([
                    A.Resize(250, 250),
                    A.RandomCrop(240, 240),
                    A.HorizontalFlip(p=0.5),
                    # A.Cutout(num_holes=8, max_h_size=32,max_w_size=32),
                    # A.ElasticTransform(),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    # A.Normalize(mean=[1.0462, 2.4636, 0.4279], std=[12.2819, 22.4455,  3.6525]),
                    ToTensorV2()
                ]),
    'valid' : A.Compose([
                    A.Resize(240, 240),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
}

# 2. 시드 고정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# 3. 모델 선언 (timm)
class EfficientNet(torch.nn.Module):
    """ Timm라이브러리에서 모델을 불러와 Classifier레이어를 추가한 클래스"""
    def __init__(self, model_name, num_classes):
        super(EfficientNet, self).__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name=model_name)

        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(in_features=n_features, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)
        print(self.model)

    def forward(self, x):
        return self.model(x)


# 4. 모델 및 학습데이터 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientNet(MODEL_NAME, num_classes=4).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)

skeleton_csv = SkeletonCSV(dataset_dir)
train_dataset, valid_dataset = train_test_split(skeleton_csv.df, test_size=0.2, random_state=42, stratify=skeleton_csv.df.to_numpy()[:,-1])

train_dataset = SkeletonDataset(train_dataset, transform=A_transforms['train'])
valid_dataset = SkeletonDataset(valid_dataset, transform=A_transforms['valid'])
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

calc_train_acc = torchmetrics.Accuracy()
calc_train_f1 = torchmetrics.F1(num_classes=4)
calc_valid_acc = torchmetrics.Accuracy()
calc_eval_acc = torchmetrics.Accuracy()

wandb.init(project="Home-Training", entity='nudago')
wandb_config = wandb.config
wandb_config.learning_rate = learning_rate
wandb_config.batch_size = batch_size
wandb_config.step_size = step_size
wandb_config.epochs = epochs


# 5. 학습
def train():
    best_loss = 999999999
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, total=train_loader.__len__(), unit="batch") as train_bar:
            for batch_idx, (inputs, labels) in enumerate(train_bar):
                example_ct = epoch * (len(train_loader)) + batch_idx
                train_bar.set_description(f"Train Epoch: {epoch}")

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()

                running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(train_loader)

                train_acc = calc_train_acc(outputs.argmax(1), labels)
                train_f1 = calc_train_f1(outputs.argmax(1), labels)

                train_bar.set_postfix(loss=epoch_loss, acc=train_acc.item(), f1=train_f1.item())
                #print(float(lr_scheduler.get_last_lr()[0]))
                wandb.log({'train_loss':loss.item(), 'train_acc':train_acc, 'learning_rate':float(lr_scheduler.get_last_lr()[0])}, step=example_ct)
        lr_scheduler.step()
        model.eval()
        running_loss=0.0
        with tqdm(valid_loader, total=valid_loader.__len__(), unit='batch') as valid_bar:
            for batch_idx, (inputs, labels) in enumerate(valid_bar):
                valid_bar.set_description(f"Valid Epoch: {epoch}")

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()

                running_loss += loss.item()# * inputs.size(0)
                epoch_loss = running_loss / len(valid_loader)

                valid_acc = calc_valid_acc(outputs.argmax(1), labels)
                valid_bar.set_postfix(loss=epoch_loss, acc=valid_acc.item())

            #if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, f'{save_dir}{MODEL_NAME}/{MODEL_NAME}_lr{learning_rate}_batch{batch_size}_epoch{epoch}_valid_loss{epoch_loss:.5f}.pt')
            early_stop_value =0
            #else:
            #    early_stop_value += 1
        wandb.log({'valid_loss':epoch_loss, 'valid_acc':calc_valid_acc.compute()}, step=example_ct+1)

# 6. 추론
def eval():
    skeleton_csv = SkeletonCSV(dataset_dir)
    eval_dataset = SkeletonDataset(skeleton_csv.test_df, transform=A_transforms['valid'])
    eval_loader = DataLoader(eval_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    best_loss = 999999999
    model.load_state_dict(torch.load(".\\results\\efficientnet_b0\\efficientnet_b0_lr0.0001_batch8_epoch59_valid_loss0.19583.pt"))
    model.eval()
    running_loss=0.0
    with tqdm(eval_loader, total=eval_loader.__len__(), unit='batch') as eval_bar:
        for batch_idx, (inputs, labels) in enumerate(eval_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs.argmax(1), labels)
            loss = criterion(outputs, labels)

            outputs = outputs.cpu().detach()
            labels = labels.cpu().detach()

            running_loss += loss.item()# * inputs.size(0)
            epoch_loss = running_loss / len(eval_loader)

            eval_acc = calc_eval_acc(outputs.argmax(1), labels)
            eval_bar.set_postfix(loss=epoch_loss, acc=eval_acc.item())
            wandb.log({'eval_loss':epoch_loss, 'eval_acc':calc_eval_acc.compute()})
    print(calc_eval_acc.compute())

if __name__ == '__main__':
    #if not os.path.exists(f'{save_dir}{MODEL_NAME}'): # save_dir폴더 생성
    #    os.makedirs(f'{save_dir}{MODEL_NAME}')
    
    #train() # 학습
    eval() # 추론