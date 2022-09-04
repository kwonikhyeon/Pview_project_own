from ctypes import sizeof
from re import L
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import skimage
from skimage.filters import meijering, sato, frangi, hessian, rank
from skimage.morphology import disk, ball
from skimage.filters.rank import otsu
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage.future import graph
from skimage.morphology import disk
from skimage.filters import frangi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy import ndimage as ndi
from skimage import color, data, filters, graph, measure, morphology
from mpl_toolkits.mplot3d import Axes3D
from skimage.util import img_as_ubyte

from skimage.morphology import diameter_closing
from skimage import data
from skimage.morphology import closing
from skimage.morphology import square
from skimage.filters import frangi, gabor
from skimage import measure, morphology


def wrinkleDetect(data_dir, imgName):
    img = cv2.imread(os.path.join(data_dir, imgName))
    img = cv2.resize(img, (256,256))
    img_blur = cv2.GaussianBlur(img, (0,0), 1.3)
    img_canny = cv2.Canny(img_blur, 50,50)
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

    result = cv2.merge(result_planes) #병합
    result_norm = cv2.merge(result_norm_planes) #병합

    img_lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab) # lab convert
    img_l, img_a, img_b = cv2.split(img_lab) # L : 밝기 / A : 초록-빨강 / B : 파랑-노랑

    img_l = img_l - 80 #히스토그램 앞으로 평행이동(너무 큰 값 살릴려고)
    alpha = 1.5 #스트레칭 비율
    img_ld = np.clip((1+alpha)*img_l - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)
    
    hist_l,bins_l = np.histogram(img_l.flatten(),256,[0,256])
    hist_ld,bins_ld = np.histogram(img_ld.flatten(),256,[0,256])
    
    ret,img_binary = cv2.threshold(img_ld, 210, 255, cv2.THRESH_BINARY_INV) # THRESH_BINARY_INV : 반전된 마스크 이미지
    img_result = cv2.erode(img_binary, np.ones((3,3), np.uint8))
    
    # # 3D 리모델링
    
    # x = np.arange(0, 256, 1)
    # y = np.arange(0, 256, 1)
    # xx, yy = np.meshgrid(x,y)
    # zz = img_l[x][y]

    # #print(xx)
    # #print(zz)
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.set_title("3D Surface Plot")
    # ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='viridis')
    # plt.show()
   
    # plt.subplot(311)
    # plt.plot(hist_l)
    # plt.subplot(312)
    # plt.plot(hist_ld)
    # plt.show()
    

    zero_count = sum([len(i[i == 0]) for i in img_result])
    imgRatio = (1 - (zero_count/(256*256)))
    print(f"주름 : {round(imgRatio,5)}")
    
    cv2.imshow('img', img)
    #cv2.imshow('norm.png', result_norm)
    #cv2.imshow('binary', img_binary)
    #cv2.imshow('canny', img_canny)
    cv2.imshow('r', img_result)
    #cv2.imshow('l', img_l)
    #cv2.imshow('ld', img_ld)
    cv2.waitKey(0)

    return round(imgRatio, 5)

def wrinkleDetect2(data_dir, imgName):
    img = cv2.imread(os.path.join(data_dir, imgName))
    mask_img = cv2.imread(os.path.join(data_dir, 'mask_' + imgName))
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (375,500))

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

    result = cv2.merge(result_planes) #병합
    result_norm = cv2.merge(result_norm_planes) #병합

    img_lab = cv2.cvtColor(result_norm, cv2.COLOR_BGR2Lab) # lab convert
    img_l, img_a, img_b = cv2.split(img_lab) # L : 밝기 / A : 초록-빨강 / B : 파랑-노랑

    img_l1 = img_l - 70 #히스토그램 앞으로 평행이동(너무 큰 값 살릴려고)
    alpha = 1.5 #스트레칭 비율
    img_ld1 = np.clip((1+alpha)*img_l1 - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)
    _, thr_ld1 = cv2.threshold(img_ld1, 200, 255, cv2.THRESH_BINARY_INV)

    # img_l2 = img_l - 90
    # img_ld2 = np.clip((1+alpha)*img_l2 - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)
    # _, thr_ld2 = cv2.threshold(img_ld2, 200, 255, cv2.THRESH_BINARY_INV)

    # thr_ld = cv2.subtract(thr_ld1, thr_ld2)

    thr_ld = thr_ld1

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = cv2.split(img_hsv)

    img_s = img_s + 30
    alpha = 1.5 #스트레칭 비율
    img_sd = np.clip((1+alpha)*img_s - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)

    #라플라시안 마스크로 경계부분 강화
    # laf_mask = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    # img_sd = cv2.filter2D(img_ld, -1, laf_mask)

    blk_size = 15        # 블럭 사이즈
    C = 5               # 차감 상수 
    thr_r = cv2.adaptiveThreshold(img_sd, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                      cv2.THRESH_BINARY_INV, blk_size, C)

    #and 연산
    rld = cv2.bitwise_and(thr_r, thr_ld)
    rld = cv2.bitwise_and(rld, mask_img)

    # cv2.imshow('mask', mask_img)
    # cv2.imshow('ld', img_ld)
    # cv2.imshow('sd', img_sd)
    # cv2.imshow('thr_ld', thr_ld)
    # cv2.imshow('thr_r', thr_r)
    # cv2.imshow('rld', rld)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    cv2.imwrite(os.path.join('./result', imgName), thr_ld)
    

def identity(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image

def wrinkleDetect3(data_dir, imgName):
    #img = plt.imread(os.path.join(data_dir, imgName))
    img = cv2.imread(os.path.join(data_dir, imgName))

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) # lab convert
    img_l, img_a, img_b = cv2.split(img_lab) # L : 밝기 / A : 초록-빨강 / B : 파랑-노랑
    blur_l = cv2.GaussianBlur(img_l, (0,0), 1)
    
    blur_l = blur_l - 50 #히스토그램 앞으로 평행이동(너무 큰 값 살릴려고)
    alpha = 1.8 #스트레칭 비율
    blur_l = np.clip((1+alpha)*blur_l - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)

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

    img = np.array(cv2.merge(result_planes)) #병합
    img_norm = cv2.merge(result_norm_planes) #병합
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img1 = img - 70 #히스토그램 앞으로 평행이동(너무 큰 값 살릴려고)
    alpha = 1.8 #스트레칭 비율
    img1 = np.clip((1+alpha)*img1 - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)
    
    laf_mask = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, laf_mask)

    kernel = np.ones((3,3), np.uint8)
    img2 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)

    a = np.clip(cv2.bitwise_and(~img2,~blur_l), 0, 255).astype(np.uint8)
    blur_a = cv2.GaussianBlur(a, (0,0), 1)
    a = cv2.filter2D(blur_a, -1, laf_mask)

    datasets = {
    'face': {'image': a,
             'figsize': (15, 7),
             'diameter': 8,
             'vis_factor': 3,
             'title': 'Text detection'}
    }

    for dataset in datasets.values():
        # image with printed letters
        image = dataset['image']
        figsize = dataset['figsize']
        diameter = dataset['diameter']

        closed = closing(image, square(diameter))

        # Again we calculate the difference to the original image.
        tophat = closed - image
        result_tophat = dataset['vis_factor'] * tophat

    ret, tophat_bin = cv2.threshold(result_tophat, 80, 255, cv2.THRESH_BINARY)

    blur = cv2.GaussianBlur(img1, (0,0), 1)
    canny = cv2.Canny(blur, 50, 150)

    

    cv2.imwrite(os.path.join('./wrinkle result', imgName), a)
    
    # cv2.imshow('0', ~img)
    # # cv2.imshow('0-1', img_norm)
    # cv2.imshow('1', ~img2)
    # cv2.imshow('2', ~blur_l)
    # cv2.imshow('3', a)
    # cv2.imshow('4', result_tophat)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def master_control(image):
    # image = cv2.resize(image, (int(image.shape[1]*0.3), int(image.shape[0]*0.3)), interpolation=cv2.INTER_CUBIC)  
    b, g, r = cv2.split(image)  # image

    sk_frangi_img = frangi(g, scale_range=(0, 1), scale_step=0.01, alpha=1.5, beta=0.01)  
    sk_frangi_img = morphology.closing(sk_frangi_img, morphology.disk(1))
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency=0.35, theta=0)
    sk_gabor_img_2, sk_gabor_2 = gabor(g, frequency=0.35, theta=45) 
    sk_gabor_img_3, sk_gabor_3 = gabor(g, frequency=0.35, theta=90)
    sk_gabor_img_4, sk_gabor_4 = gabor(g, frequency=0.35, theta=360)  
    sk_gabor_img_1 = morphology.opening(sk_gabor_img_1, morphology.disk(2))
    sk_gabor_img_2 = morphology.opening(sk_gabor_img_2, morphology.disk(1))
    sk_gabor_img_3 = morphology.opening(sk_gabor_img_3, morphology.disk(2))
    sk_gabor_img_4 = morphology.opening(sk_gabor_img_4, morphology.disk(2))
    all_img = cv2.add(0.1 * sk_gabor_img_2, 0.9 * sk_frangi_img)  # + 0.02 * sk_gabor_img_1 + 0.02 * sk_gabor_img_2 + 0.02 * sk_gabor_img_3
    all_img = morphology.closing(all_img, morphology.disk(1))
    _, all_img = cv2.threshold(all_img, 0.3, 1, 0)
    img1 = all_img
    # print(all_img, all_img.shape, type(all_img))
    # contours, image_cont = cv2.findContours(all_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # all_img = all_img + image_cont
    bool_img = all_img.astype(bool)
    label_image = measure.label(bool_img)
    count = 0

    for region in measure.regionprops(label_image):
        if region.area < 10: #   or region.area > 700
            x = region.coords
            for i in range(len(x)):
                all_img[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.98:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                all_img[x[i][0]][x[i][1]] = 0

    skel, distance = morphology.medial_axis(all_img.astype(int), return_distance=True)
    skels = morphology.closing(skel, morphology.disk(1))
    trans1 = skels 
    return skels, count  # np.uint16(skels.astype(int))


def face_wrinkle(path):
    # result = pa.curve(path, backage)
    result = cv2.imread(path)
    img, count = master_control(result)
    print(img.astype(float))
    result[img > 0.1] = 255
    cv2.imshow("result", img.astype(float))
    cv2.waitKey(0)




if __name__ == '__main__':
    data_dir1 = './face dataset'
    data_dir2 = './infinic dataset/normal'
    data_dir3 = './crop forehead dataset'
    imgName1 = 'all_kimchi.jpg'
    imgName2 = 'forehead_N_119.png'
    imgName3 = 'left_eye_kimchi.jpg'
    imgName4 = 'right_cheek_testimg.jpg'
    imgName5 = 'all_testimg2.jpg'
    imgName6 = 'N_063.png'

    # for (root, directories, files) in os.walk(data_dir2):
    #     for file in files:
    #         wrinkleDetect3(root, file)
    
    # wrinkleDetect3(data_dir2, imgName6)
    face_wrinkle(os.path.join(data_dir3, imgName2))
