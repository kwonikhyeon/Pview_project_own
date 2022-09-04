#%% 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

#%% 데이터 흑백 전환
original_data_dir = './crop dataset'
data_save_dir = './wrinkle dataset/input'
size = 560

for (root, directories, files) in os.walk(original_data_dir):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            img = cv2.resize(img, (size,size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 히스토그램 스트레칭(명암비 늘리기)
            stretched_img = img - 30 #히스토그램 앞으로 평행이동(너무 큰 값 살릴려고)
            alpha = 1.1 #스트레칭 비율
            stretched_img = np.clip((1+alpha)*stretched_img - 128*alpha, 0, 255).astype(np.uint8) 

            cv2.imwrite(os.path.join(data_save_dir, file), stretched_img)

#%% 데이터 불러오기
dir_data = './wrinkle dataset'

nx, ny = 560, 560 #이미지 크기
nframe = 99 #전체 이미지 개수

nframe_train = 81 #train set 개수
nframe_val = 9 #validation set 개수 
nframe_test = 9 # test set 개수

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)



#%% train
id_frame = np.arange(1,nframe)
np.random.shuffle(id_frame)

offset_nframe = 0
dir_input = './wrinkle dataset/input'
dir_label = './wrinkle dataset/label'

for i in range(nframe_train):
    seek_num = id_frame[i + offset_nframe]
    img_input = Image.open(os.path.join(dir_input, 'all_N_'+'{0:03d}'.format(seek_num)+'.png'))
    img_label = Image.open(os.path.join(dir_label, 'label_'+ '{0:03d}'.format(seek_num)+'.png'))

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_'+'{0:03d}'.format(i)+'.npy'), label_)
    np.save(os.path.join(dir_save_train, 'input_'+'{0:03d}'.format(i)+'.npy'), input_)
# %% val
offset_nframe = nframe_train
dir_input = './wrinkle dataset/input'
dir_label = './wrinkle dataset/label'

for i in range(nframe_val):
    seek_num = id_frame[i + offset_nframe]
    img_input = Image.open(os.path.join(dir_input, 'all_N_'+'{0:03d}'.format(seek_num)+'.png'))
    img_label = Image.open(os.path.join(dir_label, 'label_'+ '{0:03d}'.format(seek_num)+'.png'))

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_'+'{0:03d}'.format(i)+'.npy'), label_)
    np.save(os.path.join(dir_save_val, 'input_'+'{0:03d}'.format(i)+'.npy'), input_)
# %% test
offset_nframe = nframe_val
dir_input = './wrinkle dataset/input'
dir_label = './wrinkle dataset/label'

for i in range(nframe_test):
    seek_num = id_frame[i + offset_nframe]
    img_input = Image.open(os.path.join(dir_input, 'all_N_'+'{0:03d}'.format(seek_num)+'.png'))
    img_label = Image.open(os.path.join(dir_label, 'label_'+ '{0:03d}'.format(seek_num)+'.png'))

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_'+'{0:03d}'.format(i)+'.npy'), label_)
    np.save(os.path.join(dir_save_test, 'input_'+'{0:03d}'.format(i)+'.npy'), input_)
# %% data check
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()
