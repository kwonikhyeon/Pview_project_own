# from pih_detector_global import pih_detector_global
from skin_cropper_global import skin_cropper
# from oil_detector_global import oil_detector
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def saturate_contrastB(Img, boundary):
    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h1, s1, v1 = cv2.split(hsv1)
    v1 = cv2.calcHist([v1],[0],None,[256],[0,256])
    

    I = printContrast(Img)
    Img = Img+(boundary - I)
    Img = np.clip(Img, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v= cv2.calcHist([v],[0],None,[256],[0,256])
    plt.plot(v, "-r")
    plt.plot(v1, "--g")
    plt.yscale('log')
    plt.show()

    # print(f"{I} -> {printContrast(Img)}")
    return Img

# 이미지 명도 출력
def printContrast(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    sort_v = np.sort(v.flatten())
    mask = np.where(sort_v!=0)

    mask_sort_v = sort_v[mask]

    cut_v = mask_sort_v[int(width*height*0.05):int(width*height*0.95)]
    return np.median(cut_v)

if __name__=="__main__":
    height = 720 # 1440
    width = 540 # 1080
    data_dir ="./infinic dataset/normal"
    imgName = "N_001.png"
    
    # 사진 크기 리사이즈
    img = cv2.resize(cv2.imread(f"{data_dir}/{imgName}", cv2.IMREAD_COLOR), (width, height))
    
    mask, masked_img = skin_cropper(img, 'all', 'B')
    
    # 밝기 보정
    # img = saturate_contrastB(masked_img, 160)
    
    # 출력부
    ## 마스크이미지
    cv2.imshow('hi', img)
    cv2.imshow('result', mask)
    cv2.imshow('result2', masked_img)

    # 일반진단테스트

    ## 유분
    # oil_detector(masked_img, True)
    
    # # ## 색소침착
    # pih_detector_global(masked_img, False)

    cv2.waitKey(0)
    cv2.destroyAllWindows()