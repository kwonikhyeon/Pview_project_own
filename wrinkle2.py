import sys
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.filters import frangi, gabor
from skimage import measure, morphology


def display_image(image, name):
    window_name = name
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def master_control(image):
    # image = cv2.resize(image, (int(image.shape[1]*0.3), int(image.shape[0]*0.3)), interpolation=cv2.INTER_CUBIC)  
    v, g, r = cv2.split(image)  # image

    c2 = v - 50 #히스토그램 앞으로 평행이동(너무 큰 값 살릴려고)
    alpha = 1.5 #스트레칭 비율
    c2 = np.clip((1+alpha)*c2 - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)

    canny1 = cv2.Canny(c2, 100,30)
    display_image(c2, 'c2')
    display_image(canny1, 'canny')

    sk_frangi_img = canny1
    # frangi(g, scale_range=(1, 1.5), scale_step=0.1) #alpha=1.5, beta=0.01  
    
    # sk_frangi_img = morphology.closing(sk_frangi_img, morphology.disk(1))
    # display_image(sk_frangi_img, "sk_frangi_img")
    sk_gabor_img_1, sk_gabor_1 = gabor(c2, frequency=0.25, theta=0)
    sk_gabor_img_2, sk_gabor_2 = gabor(c2, frequency=0.25, theta=45) 
    sk_gabor_img_3, sk_gabor_3 = gabor(c2, frequency=0.25, theta=90)
    sk_gabor_img_4, sk_gabor_4 = gabor(c2, frequency=0.25, theta=360)  
    
    sk_gabor_img_1 = morphology.opening(sk_gabor_img_1, morphology.disk(2))
    sk_gabor_img_2 = morphology.opening(sk_gabor_img_2, morphology.disk(1))
    sk_gabor_img_3 = morphology.opening(sk_gabor_img_3, morphology.disk(2))
    sk_gabor_img_4 = morphology.opening(sk_gabor_img_4, morphology.disk(2))
    all_img = cv2.add(0.02 * sk_gabor_img_1 + 0.12 * sk_gabor_img_2 + 0.12 * sk_gabor_img_3 + 0.04 * sk_gabor_img_4, 0.7 * sk_frangi_img)  # + 0.02 * sk_gabor_img_1 + 0.02 * sk_gabor_img_2 + 0.02 * sk_gabor_img_3 + 0.04 * sk_gabor_img_4
    
    all_img = morphology.closing(all_img, morphology.disk(1))

    display_image(sk_frangi_img, "sk_frangi_img")
    display_image(sk_gabor_img_1, "sk_gabor_img_1")
    display_image(sk_gabor_img_2, "sk_gabor_img_2")
    display_image(sk_gabor_img_3, "sk_gabor_img_3")
    display_image(sk_gabor_img_4, "sk_gabor_img_4")
    display_image(all_img, "all_img")

    
    # _, all_img = cv2.threshold(all_img, 0.3, 1, 0)
    
    # print(all_img, all_img.shape, type(all_img))
    
    display_image(image, "img")

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

def face_wrinkle2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)  # image

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #테두리 생성
    border = np.zeros(np.shape(img))
    contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(border, contours, 0, (255,255,255), 3, cv2.LINE_8, hier)
    display_image(border, 'border')

    #히스토그램 스트레칭
    c2 = v - 70 
    alpha = 1.8 
    c2 = np.clip((1+alpha)*c2 - 128*alpha, 0, 255).astype(np.uint8)
    c2 = ~c2

    #-------------------------------------------------------
    robertsx = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    robertsy = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
    robertsX = cv2.filter2D(c2, cv2.CV_64F, robertsx)
    robertsY = cv2.filter2D(c2, cv2.CV_64F, robertsy)
    prewittx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitty = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_ = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])

    prewittX = cv2.filter2D(c2, -1, prewittx)
    prewittY = cv2.filter2D(c2, -1, prewitty)
    prewitt_ = cv2.filter2D(c2, -1, prewitt_)
    sobelX = cv2.Sobel (c2, -1, 1, 0, ksize=3)
    sobelY = cv2.Sobel (c2, -1, 0, 1, ksize=3)
    sobel = sobelX + sobelY
    scharrX = cv2.Sobel (c2, -1, 1, 0, ksize = cv2.FILTER_SCHARR)
    scharrY = cv2.Sobel (c2, -1, 0, 1, ksize = -1)
    scharr = scharrX + scharrY

    titles = ['original', 'roberts-X', 'roberts-Y', 'prewitt-X', 'prewitt-Y', 'prewitt_', 'sobel-X', 'sobel-Y', 'sobel', 'scharr-X', 'scharr-Y', 'scharr']
    images = [img, robertsX, robertsY, prewittX, prewittY, prewitt_, sobelX, sobelY, sobel, scharrX, scharrY, scharr]
    cv2. imshow('original', img)
    cv2. imshow('Prewitt',prewitt_)
    
    cv2.imshow('Sobel', sobel)
    cv2. imshow('Scharr', scharr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.figure(figsize = (12, 12))
    for i in range(12):
        plt.subplot(4, 3, i+1)
        plt. imshow(images [i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


#---------------------------------------------------------------------



    blk_size = 9
    C = 5 
    ret, th1 = cv2.threshold(c2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # ---② 어뎁티드 쓰레시홀드를 평균과 가우시안 분포로 각각 적용
    th2 = cv2.adaptiveThreshold(c2, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                        cv2.THRESH_BINARY_INV, blk_size, C)
    th3 = cv2.adaptiveThreshold(c2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY_INV, blk_size, C)

    th = cv2.bitwise_and(th2, th3)
    th = cv2.bitwise_and(th, th1)
    th = th1 - th


    # display_image(th1, 'th1')
    # display_image(th2, 'th2')
    # display_image(th3, 'th3')
    # display_image(th, 'th')

    # c2 = cv2.GaussianBlur(c2, (0,0), 2)
    canny1 = cv2.Canny(c2, 10,30)

    sk_frangi_img = c2 #, scale_range=(1, 1.5), scale_step=0.1     alpha=1.5, beta=0.01  
    display_image(sk_frangi_img, 'f')
    sk_gabor_img_1, sk_gabor_1 = gabor(sk_frangi_img, frequency=0.25, theta=0)
    sk_gabor_img_2, sk_gabor_2 = gabor(sk_frangi_img, frequency=0.25, theta=45) 
    sk_gabor_img_3, sk_gabor_3 = gabor(sk_frangi_img, frequency=0.25, theta=80)
    sk_gabor_img_4, sk_gabor_4 = gabor(sk_frangi_img, frequency=0.25, theta=200) 

    sk_gabor_img_1 = sk_gabor_img_1.astype(np.uint8) - border 
    sk_gabor_img_2 = sk_gabor_img_2.astype(np.uint8) - border 
    sk_gabor_img_3 = sk_gabor_img_3.astype(np.uint8) - border 
    sk_gabor_img_4 = sk_gabor_img_4.astype(np.uint8) - border 
    
    # sk_gabor_img_1 = morphology.opening(sk_gabor_img_1, morphology.disk(2))
    # sk_gabor_img_2 = morphology.opening(sk_gabor_img_2, morphology.disk(1))
    # sk_gabor_img_3 = morphology.opening(sk_gabor_img_3, morphology.disk(2))
    # sk_gabor_img_4 = morphology.opening(sk_gabor_img_4, morphology.disk(2))
    all_img = sk_gabor_img_1 + sk_gabor_img_2 + sk_gabor_img_3 + sk_gabor_img_4
    # cv2.add(0.2 * sk_gabor_img_1 + 0.2 * sk_gabor_img_2 + 0.2 * sk_gabor_img_3 + 0.2 * sk_gabor_img_4, 0.02 * sk_frangi_img)  # + 0.02 * sk_gabor_img_1 + 0.02 * sk_gabor_img_2 + 0.02 * sk_gabor_img_3 + 0.04 * sk_gabor_img_4
    
    all_img = all_img - border
    
    # all_img = morphology.closing(all_img, morphology.disk(1))
    
    
    display_image(sk_gabor_1, '1')    
    display_image(sk_gabor_2, '2')
    display_image(sk_gabor_3, '3')
    display_image(sk_gabor_4, '4')

    cv2.imshow('img', c2)
    cv2.imshow('result', all_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    data_dir1 = './face dataset'
    data_dir2 = './infinic dataset/normal'
    data_dir3 = './part dataset'
    imgName1 = 'all_kimchi.jpg'
    imgName2 = 'forehead_N_063.png'
    imgName3 = 'left_eye_N_073.png'
    imgName4 = 'right_cheek_testimg.jpg'
    imgName5 = 'all_testimg2.jpg'
    imgName6 = 'all_N_067.png'

    # for (root, directories, files) in os.walk(data_dir2):
    #     for file in files:
    #         wrinkleDetect3(root, file)
    
    # wrinkleDetect3(data_dir2, imgName6)
    face_wrinkle2(os.path.join(data_dir3, imgName3))