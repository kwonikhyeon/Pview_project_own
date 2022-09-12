# 1. 밝기 균등
# (2. 뾰루지나 털 등 기타 색 제거) -> 아직 안함
# 3. 평균 피부 색깔 구하기
# 4. Fitzpatrick Scale에서 어느 타입인지 비교
from __future__ import division
from distutils.log import error
from posixpath import split
from turtle import back
import numpy as np
import os
import cv2
from numpy.core.fromnumeric import size
from matplotlib import pyplot as plt

def display_image(image, name):
    window_name = name
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


# segment using otsu binarization and thresholding
def segment_otsu(image_grayscale, img_BGR):
    threshold_value, threshold_image = cv2.threshold(image_grayscale, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    display_image(threshold_image, "otsu") 
    threshold_image_binary = 1- threshold_image/255
    threshold_image_binary = np.repeat(threshold_image_binary[:, :, np.newaxis], 3, axis=2)
    img_face_only = np.multiply(img_BGR, threshold_image_binary)
    return img_face_only


def global_skinToneDetect(img):
    image_path = './crop dataset/all_N_001.png'
    img_BGR = img

    img_face_only = img_BGR
    img_HSV = cv2.cvtColor(img_face_only, cv2.COLOR_BGR2HSV)
    img_YCrCb = cv2.cvtColor(img_face_only, cv2.COLOR_BGR2YCrCb)
    display_image(img_HSV, "HSV")
    display_image(img_YCrCb, "YCrCb")

    # aggregate skin pixels
    blue = []
    green = []
    red = []

    height, width, channels = img_face_only.shape

    for i in range (height):
        for j in range (width):
            if((img_HSV.item(i, j, 0) <= 170) and (140 <= img_YCrCb.item(i, j, 1) <= 170) and (90 <= img_YCrCb.item(i, j, 2) <= 120)):
                blue.append(img_face_only[i, j].item(0))
                green.append(img_face_only[i, j].item(1))
                red.append(img_face_only[i, j].item(2))
            else:
                img_face_only[i, j] = [0, 0, 0]

    display_image(img_face_only, "final segmentation")

    # determine mean skin tone estimate
    skin_tone_estimate_BGR = [np.mean(blue), np.mean(green), np.mean(red)]
    avr_color = cv2.merge([np.full((560,560), skin_tone_estimate_BGR[0], dtype=np.uint8),
                np.full((560,560), skin_tone_estimate_BGR[1], dtype=np.uint8),
                np.full((560,560), skin_tone_estimate_BGR[2], dtype=np.uint8)])
    display_image(avr_color, "avr")
    print("mean skin tone estimate (BGR)", skin_tone_estimate_BGR[0], skin_tone_estimate_BGR[1], skin_tone_estimate_BGR[2], "]")
    return skin_tone_estimate_BGR

def skinToneDetect(data_dir, imgName):

    # 피부톤 컬러차트
    colorChart = [  (0,0,0),         # 0. 점수 없음
                    (212,230,248),  # 1. Ecru (가장 밝음)
                    (195,220,245),  # 2. Beige
                    (187,214,243),  # 3. Moccasin
                    (161,198,239),  # 4. Fawn
                    (144,188,236),  # 5. Tan
                    (119,172,231),  # 6. Wren
                    (93,157,227),   # 7. Cinnamon
                    (76,146,224),   # 8. Tawny
                    (59,136,221),   # 9. Nutmeg
                    (35,116,205),   # 10. Copper
                    (26,87,154),    # 11. Woodland
                    (16,53,94)]     # 12. Teak
    
    # 이미지 입력
    img = cv2.imread(os.path.join(data_dir, imgName))
    imgSize = 560

    gridValue = []

    division = 40 #560 픽셀을 나눌 기준 픽셀 수
    for mode in (0,1):
        for i, startValue in enumerate(range(0,imgSize,division), start=1):
            if mode == 0:
                split_img = img[startValue:i*division, :]
            elif mode == 1:
                split_img = img[:, startValue:i*division]

            img_b, img_g, img_r = cv2.split(split_img)   
            
            if len(img_b[img_b>0]) == 0:
                filtered_img_b = [0]
            else:
                filtered_img_b = img_b[img_b>0]
            if len(img_g[img_g>0]) == 0:
                filtered_img_g = [0]
            else:
                filtered_img_g = img_g[img_g>0]
            if len(img_r[img_r>0]) == 0:
                filtered_img_r = [0]
            else:
                filtered_img_r = img_r[img_r>0]

            rgb_value = [np.mean(filtered_img_b), np.mean(filtered_img_g), np.mean(filtered_img_r)]
            
            errorRate = []
            for tone in colorChart:
                errorRate.append(sum(np.abs(np.array(rgb_value) - np.array(tone))))
            partValue = errorRate.index(min(errorRate))
            gridValue.append(partValue)
        # print('done')

    
    print(imgName, ' Skin Score : ', '[',round(sum(np.array(gridValue)*0.29761), 4), ']')
    





# if __name__ == '__main__':
#     data_dir = './crop dataset'
#     imgName = 'all_N_221.png'
#     skinToneDetect(data_dir, imgName)
#     skinToneDetect(data_dir, 'all_white.jpg')
#     skinToneDetect(data_dir, 'all_black.jpg')
    
