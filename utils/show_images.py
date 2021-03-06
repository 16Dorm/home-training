import matplotlib.pyplot as plt 
import pandas as pd
import cv2


select_video = 'pushup_00'


df = pd.read_csv('./train/train.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'images', 'label'])

# 필요한 데이터만 변수에 넣음
img_name = df['images']
img_label = df['label']

# for문을 편하게 돌기 위해 zip으로 묶음
images_and_labels = list(zip(img_name, img_label))

# for문을 돌면서 필요한 변수들 선언
pre_img_name = img_name[0][:22]
start_index = 0
end_index = 30
df_len = len(df)
is_end = False

# 원하는 영상이 있는지 확인
while (select_video != img_name[start_index][14:23]):
    start_index += 1
    if (start_index >= df_len):
        print('no video')
        exit()

pre_img_name = img_name[start_index][:23]
end_index = start_index + 30

if (end_index >= df_len):
    end_index = df_len-1

# 모든 cvs에 저장된 이미지 출력
while(start_index < df_len):
    fig = plt.figure(figsize=(15, 10)) 
    for index, (img_path, label) in enumerate(images_and_labels[start_index:end_index]):
        # 영상 이름이 바뀌면 멈추도록함
        if(pre_img_name != img_path[:23]):
            is_end = True
            pre_img_name = img_path[:23]
            break
        
        # 제목, 위치 등 설정
        plt.suptitle(img_path[14:23],fontsize=25)
        plt.subplot(6, 5, index + 1)
        plt.axis('off')

        # 마지막 영상 이름 update
        pre_img_name = img_path[:23]

        # 이미지 불러오고 라벨링 값과 함께 나타내기
        img = cv2.imread(img_path)
        plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.subplots_adjust(wspace=0.8)
        plt.title('label: %i' % label, fontsize=12)
    
    # 출력
    plt.show()

    # 특정 영상의 마지막인지 아닌지 확인
    if(not(is_end)):
        start_index += 30
        end_index += 30
    else:
        is_end = False
        start_index += 1
        end_index += 1
        
