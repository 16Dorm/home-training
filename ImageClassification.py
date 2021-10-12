import numpy as np
import pandas as pd
import torch

from data_loader.SkeletonDataset import SkeletonDataset
from data_loader.SkeletonDataLoader import SkeletonDataLoader

import torchmetrics
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
# import wandb
# wandb.login()

# 1. 데이터 경로 및 하이퍼파라미터
dataset_dir = './'
MODEL_NAME = "efficientnet_b0"
learning_rate = 2e-5
batch_size = 64
step_size = 5
epochs = 20
early_stop =5

A_transforms = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])

# 2. 데이터 로더
class Load_CSV():
    """CSV데이터를 불러와 """
    def __init__(self, dir):
        self.df = pd.read_csv(dir, names=["head", "shoulder", "elbow", "hand", "hip", "foot", "elbow_angle", "hip_angle", "knee_angle", "image_path", "label"])
        self.df = self.df[['image_path','label']]


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
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

dataloader = SkeletonDataLoader(dataset_dir, batch_size=64, trsfm=A_transforms)
train_loader, valid_loader = dataloader.split_validation()

calc_train_acc = torchmetrics.Accuracy()
calc_train_f1 = torchmetrics.F1(num_classes=4)
calc_valid_acc = torchmetrics.Accuracy()


# wandb.init(project="Home-Training", entity='nudago')
# wandb_config = wandb.config
# wandb_config.learning_rate = learning_rate
# wandb_config.batch_size = batch_size
# wandb_config.step_size = step_size
# wandb_config.epochs = epochs


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

            train_bar.set_postfix(loss=epoch_loss, acc=train_acc, f1=train_f1)
            # wandb.log({'train_loss':loss.item(), 'train_acc':train_acc}, step=example_ct)
    lr_scheduler.step()

    model.eval()
    running_loss=0.0
    with tqdm(valid_loader, total=valid_loader.__len__(), unit='batch') as valid_bar:
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            valid_bar.set_description(f"Vrain Epoch: {epoch}")

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            outputs = outputs.cpu().detach()
            labels = labels.cpu().detach()

            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader)

            valid_acc = calc_valid_acc(outputs.argmax(1), labels)
            valid_bar.set_postfix(loss=epoch_loss, acc=valid_acc.compute())
    # wandb.log({'valid_loss':epoch_loss, 'train_acc':calc_valid_acc.compute()}, step=example_ct)


# 6. 추론5
if __name__ == "__main__":
    #df_train = Load_CSV(train_dir)
    pass