"""
모델의 아키텍처 (3개 층)

# 1번 레이어 : 합성곱층(Convolutional layer)
합성곱(in_channel = 1, out_channel = 32, kernel_size=3, stride=1, padding=1) + 활성화 함수 ReLU
맥스풀링(kernel_size=2, stride=2))

# 2번 레이어 : 합성곱층(Convolutional layer)
합성곱(in_channel = 32, out_channel = 64, kernel_size=3, stride=1, padding=1) + 활성화 함수 ReLU
맥스풀링(kernel_size=2, stride=2))

# 3번 레이어 : 전결합층(Fully-Connected layer)
특성맵을 펼친다. # batch_size × 7 × 7 × 64 → batch_size × 3136
전결합층(뉴런 10개) + 활성화 함수 Softmax
"""

import torch
import torch.nn as nn

#임의의 텐서 만들기 (크기 1 1 28 28)
inputs = torch.Tensor(1,1,28,28)
print('텐서의 크기 : {}'.format(inputs.shape))

# 합섭곱층과 풀링 선언
# 1. 1채널 입력받아 32채널 뽑고 커널사이즈 3 패딩 1
conv1 = nn.Conv2d(1,32,3, padding=1)
print('1번레이어 :', conv1)

# 2. 두번쨰 합성곱층 구현
# 32채널 입력 64채널 뽑고 커널사이즈 3 패딩 1
conv2 = nn.Conv2d(32,64,3, padding=1)
print('2번레이어 :', conv2)

# 3.맥스풀링 구현
# 정수 하나를 인자로 넣으면 커널 사이즈와 스트라이드가 둘다 해당값으로 지정
pool = nn.MaxPool2d(2)
print('맥스풀링 :', pool)




# 구현체 연결하여 모델 만들기
# 1. 아직까진 선언만한거고 연결 X, 1번, 2번 레이어 통과 후 텐서크기 확인, 맥스풀링까지 통과한 후 텐서크기 확인
out = conv1(inputs)
print('1번레이어 통과 후 텐서 크기 :', out.shape)
out = pool(out)
print('맥스풀링 통과 후 텐서 크기 :', out.shape)

out = conv2(out)
print('2번레이어 통과 후 텐서 크기 :', out.shape)
out = pool(out)
print('맥스풀링 통과 후 텐서 크기 :', out.shape)

# 2. 텐서 펼치기
# .view()이용하혀 펼치기 size(0)-> 첫번쨰 차원 / size(1) -> 두번쨰 차원
out = out.view(out.size(0), -1) #첫번쨰 차원인 배치 차원은 그대로두고 나머지는 펼쳐라
print('첫번째 차원 그대로 나머지 차원 펼친 후 :',out.shape)

# 전 결합층을 통과시키기, 출력층으로 10개 뉴런 배치해 10개 차원의 텐서로 변환
fc = nn.Linear(3136,10) #input_dim = 3136 output_dim = 10
out = fc(out)
print('전 결합층 통과 후 10개 차원 텐서 변환 :',out.shape)\



# CNN으로 MNIST 분류하기
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

# 1. GPU 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1-1. 랜덤 시드 고정 (무슨말이지?)
torch.manual_seed(777)

# 1-2. GPU 사용 가능일 경우 랜덤 시드 고정
if device =='cuda':
    torch.cuda.manual_seed_all(777)

# 2. 학습에 사용 할 파라미터 설정 / 데이터로더 사용하기 위한 데이터셋 정의 / 데이터 로더 사용해 배치크기 지정
learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data', #다운로드 경로 지정
                          train=True, # True 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), #텐서로 변환
                          download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로지정
                         train=False, # False는 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), #텐서로 변환
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)



# 클래스로 모델 설겨
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째 층
        # ImageIn shape = (?, 28, 28, 1)
        # Conv -> (? ,28, 28, 32)
        # Pool -> (? ,14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        # 두번째 층
        # ImageIn shape = (?, 14, 14, 32)
        # Conv -> (?, 14, 14, 64)
        # Pool -> (?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7*7*64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) #전 결합층을 위한 Flatten(다 펼치기)
        out = self.fc(out)
        return out

# CNN 모델 정의 / 비용함수, 옵티마이저 정의
model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device) #비용 함수에 소프트맥스 함수 포함되어져있음
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))




# 훈련시키기
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))



# 테스트
with torch.no_grad(): # 학습 진행 하지 않고 바로 테스트
    X_test = mnist_test.test_data.view(len(mnist_test),1,28,28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
