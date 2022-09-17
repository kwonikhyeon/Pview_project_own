# 붉은부분이나 점 등 제거
# 밝기 균일화
# 모공'만'검출

import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


def poreDetect(data_dir, imgName):
    img = cv2.imread(os.path.join(data_dir, imgName))
    # mask_img = cv2.imread(os.path.join(data_dir, 'mask_' + imgName))
    # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8)) #RGB 각 채널을 대상으로 커널만큼 팽창시킴
        bg_img = cv2.medianBlur(dilated_img, 21) #블러
        diff_img = 255 - cv2.absdiff(plane, bg_img) #배경제거(그림자로 인식된 부분 제거)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) #표준화
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        # cv2.imwrite(os.path.join('./test', '1_' + imgName), dilated_img)
        # cv2.imwrite(os.path.join('./test', '2_' + imgName), bg_img)
        # cv2.imwrite(os.path.join('./test', '3_' + imgName), cv2.absdiff(plane, bg_img))

    result = cv2.merge(result_planes) #병합
    result_norm = cv2.merge(result_norm_planes) #병합

    img_lab = cv2.cvtColor(result_norm, cv2.COLOR_BGR2Lab) # lab convert
    img_l, img_a, img_b = cv2.split(img_lab) # L : 밝기 / A : 초록-빨강 / B : 파랑-노랑

    img_l1 = img_l - 70 #히스토그램 앞으로 평행이동(너무 큰 값 살릴려고)
    alpha = 1.5 #스트레칭 비율
    img_ld1 = np.clip((1+alpha)*img_l1 - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)
    _, thr_ld1 = cv2.threshold(img_ld1, 200, 255, cv2.THRESH_BINARY_INV)

    thr_ld = thr_ld1
    
    # cv2.imwrite(os.path.join('./pore result', imgName), thr_ld)
    # cv2.imwrite(os.path.join('./test', 'result_' + imgName), result)
    # cv2.imwrite(os.path.join('./test', 'result_norm_' + imgName), result)

if __name__ == '__main__':
    data_dir1 = './face dataset'
    data_dir2 = './crop dataset'
    imgName1 = 'left_cheek_kimchi.jpg'
    imgName2 = 'forehead_testimg.jpg'
    imgName3 = 'left_eye_kimchi.jpg'
    imgName4 = 'right_cheek_testimg.jpg'
    imgName5 = 'all_N_221.png'

    # for (root, directories, files) in os.walk(data_dir2):
    #     for file in files:
    #         poreDetect(data_dir2, file)

    poreDetect(data_dir2, imgName5)
