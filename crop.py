# input image crop code
from tkinter.messagebox import NO
from cv2 import FONT_HERSHEY_SIMPLEX
import mediapipe as mp
import cv2
import csv
import os, sys
import numpy as np
import time
import matplotlib.pyplot as plt

#1080*1440 : 핸드폰 카메라로 찍혀서 들어오는 이미지 사이즈
WIDTH = 1080
HEIGHT = 1440
WIDTH_WRAPING = 560
HEIGHT_WRAPING = 560

# Separate Face Part
right_cheek = [350, 277, 371, 423, 322, 273, 424, 378, 379, 416, 433, 376, 352, 346, 347, 348, 349]
left_cheek = [121, 47, 142, 203, 92, 43, 204, 149, 150, 192, 213, 147, 123, 117, 118, 119, 120]
right_eye = [446, 342, 353, 383, 372, 340, 346, 347, 348, 349, 452, 341, 256, 252, 253, 254, 339]
left_eye = [226, 113, 124, 156, 143, 111, 117, 118, 119, 120, 232, 112, 26, 22, 23, 24, 110]
forehead = [10, 338, 297, 332, 284, 334, 296, 336, 9, 107, 66, 105, 54, 103, 67, 109]
chin = [17, 406, 422, 430, 379, 400, 152, 176, 150, 210, 202, 182]
nose = [6, 122, 188, 217, 198, 209, 49, 48, 219, 218, 237, 44, 1, 274, 457, 438, 439, 278, 279, 429, 420, 437, 412, 351]


def create_landmark_img(data_dir=None, imgName=None, outputType=None, inputImg=None):
    
    img = cv2.imread(os.path.join(data_dir, imgName))
    img = cv2.resize(img, (WIDTH,HEIGHT))

    mp_drawing = mp.solutions.drawing_utils # drawing helpers
    mp_holistic = mp.solutions.holistic # mediapipe solutions
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

     # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Recolor Feed
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image = cv2.flip(image, 1)
        image.flags.writeable = False
        
        # Make Detections
        # results = holistic.process(image)
        results = face_mesh.process(image)

        weight = 1.0
        
        if results.multi_face_landmarks:
            for facial_landmarks in results.multi_face_landmarks:
                # Total 468
                # for i in range(0, 468):
                #     pt1 = facial_landmarks.landmark[i]
                #     x = int(pt1.x * WIDTH * 1.0)
                #     y = int(pt1.y * HEIGHT * weight)

                #     cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
                #     cv2.putText(image, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                
                # for i in range(0, 200):
                #     pt1 = facial_landmarks.landmark[i]
                #     x = int(pt1.x * WIDTH)
                #     y = int(pt1.y * HEIGHT * weight)

                #     cv2.circle(image, (x, y), 3, (0, 0, 0), 1)
                    
                for i in forehead:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * WIDTH)
                    y = int(pt1.y * HEIGHT * weight)

                    cv2.circle(image, (x, y), 10, (0, 155, 0), -1)
                    #cv2.putText(image, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                for i in right_eye:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * WIDTH)
                    y = int(pt1.y * HEIGHT * weight)

                    cv2.circle(image, (x, y), 10, (0, 155, 155), -1)
                    #cv2.putText(image, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                for i in left_eye:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * WIDTH)
                    y = int(pt1.y * HEIGHT * weight)

                    cv2.circle(image, (x, y), 10, (0, 155, 155), -1)
                    #cv2.putText(image, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                for i in left_cheek:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * WIDTH)
                    y = int(pt1.y * HEIGHT * weight)

                    cv2.circle(image, (x, y), 10, (155, 0, 0), -1)
                    #cv2.putText(image, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                for i in right_cheek:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * WIDTH)
                    y = int(pt1.y * HEIGHT * weight)

                    cv2.circle(image, (x, y), 10, (155, 0, 0), -1)
                    #cv2.putText(image, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                for i in nose:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * WIDTH)
                    y = int(pt1.y * HEIGHT * weight)

                    cv2.circle(image, (x, y), 10, (255, 255, 255), -1)
                    #cv2.putText(image, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                
            #generate warpPerspectiveAry
            #result_r_cheek1 = warping(image, facial_landmarks, WIDTH_WRAPING, HEIGHT_WRAPING, [349, 261, 352, 266])

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if outputType == 0:
                cv2.imshow('test1', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            elif outputType == 1:
                cv2.imwrite(os.path.join('./landmark dataset', imgName), image)   

            elif outputType == 2:
                lendmark_output = []
                # Total 468
                for i in range(0, 468):
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * WIDTH)
                    y = int(pt1.y * HEIGHT * weight)
                    lendmark_output.append([x,y])
                return lendmark_output

def warping(image, facial_landmarks, WIDTH_WRAPING, HEIGHT_WRAPING, points):
    warpPerspectiveAry = []
    warpPerspectiveAry.append(points[0])
    warpPerspectiveAry.append(points[1])
    warpPerspectiveAry.append(points[2])
    warpPerspectiveAry.append(points[3])

    src = np.array([warpPerspectiveAry[0], warpPerspectiveAry[1], warpPerspectiveAry[2], warpPerspectiveAry[3]], dtype = np.float32)
    dst = np.array([[0, 0], [WIDTH_WRAPING, 0], [WIDTH_WRAPING, HEIGHT_WRAPING], [0, HEIGHT_WRAPING]], dtype = np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst) # src2dst 하기 위한변형 행렬
    result = cv2.warpPerspective(image, matrix, (WIDTH_WRAPING, HEIGHT_WRAPING)) # 변형

    #cv2.resize(image, dsize=(WIDTH_WRAPING, HEIGHT_WRAPING))
    return result

def cropImage(data_dir=None, imgName=None, area='all', outputType = 0, inputImg = None):
   
    img = cv2.imread(os.path.join(data_dir, imgName))
        
    img = cv2.resize(img, (WIDTH,HEIGHT))
    mask = np.zeros((HEIGHT,WIDTH, 3),dtype = np.uint8)
    facial_landmark = create_landmark_img(data_dir, imgName, 2, inputImg=img)

    right_cheek_poly = []
    left_cheek_poly = []
    right_eye_poly = []
    left_eye_poly = []
    forehead_poly = []
    mouse_poly = []
    nose_poly = []

    if area == 'right_cheek':
        for i in right_cheek:
            right_cheek_poly.append(facial_landmark[i]) 
        poly_area = np.array(right_cheek_poly, np.int32)
        
    elif area == 'left_cheek':
        for i in left_cheek:
            left_cheek_poly.append(facial_landmark[i]) 
        poly_area = np.array(left_cheek_poly, np.int32)
    
    elif area == 'right_eye':
        for i in right_eye:
            right_eye_poly.append(facial_landmark[i]) 
        poly_area = np.array(right_eye_poly, np.int32)
        
    elif area == 'left_eye':
        for i in left_eye:
            left_eye_poly.append(facial_landmark[i]) 
        poly_area = np.array(left_eye_poly, np.int32)
        
    elif area == 'forehead':
        for i in forehead:
            forehead_poly.append(facial_landmark[i]) 
        poly_area = np.array(forehead_poly, np.int32)

    elif area == 'nose':
        for i in nose:
            nose_poly.append(facial_landmark[i]) 
        poly_area = np.array(nose_poly, np.int32)

    elif area == 'all':
        for i in right_cheek:
            right_cheek_poly.append(facial_landmark[i]) 
        poly_area1 = np.array(right_cheek_poly, np.int32)

        for i in left_cheek:
            left_cheek_poly.append(facial_landmark[i]) 
        poly_area2 = np.array(left_cheek_poly, np.int32)

        for i in right_eye:
            right_eye_poly.append(facial_landmark[i]) 
        poly_area3 = np.array(right_eye_poly, np.int32)

        for i in left_eye:
            left_eye_poly.append(facial_landmark[i]) 
        poly_area4 = np.array(left_eye_poly, np.int32)

        for i in forehead:
            forehead_poly.append(facial_landmark[i]) 
        poly_area5 = np.array(forehead_poly, np.int32)

        for i in nose:
            nose_poly.append(facial_landmark[i]) 
        poly_area6 = np.array(nose_poly, np.int32)
        

    if area == 'all':
        mask = cv2.fillPoly(mask, [poly_area1,
                                    poly_area2,
                                    poly_area3,
                                    poly_area4,
                                    poly_area5,
                                    poly_area6], (255,255,255))

        maxPoint = [WIDTH, HEIGHT]
        minPoint = [0,0]

    else:
        mask = cv2.fillPoly(mask, [poly_area], (255,255,255))
        maxPoint = np.apply_along_axis(lambda a: np.max(a), 0, poly_area)
        minPoint = np.apply_along_axis(lambda a: np.min(a), 0, poly_area)

    masked_img = cv2.bitwise_and(img,mask)
    #masked_img = cv2.flip(masked_img, 1)

    cut_img = masked_img[minPoint[1]:maxPoint[1], minPoint[0]:maxPoint[0]]
    masked_cut_img = mask[minPoint[1]:maxPoint[1], minPoint[0]:maxPoint[0]]

    if area == 'all':
        warp_point = [[facial_landmark[123][0], facial_landmark[10][1]], # 좌측볼 x좌표 + 이마 끝 y좌표
                        [facial_landmark[352][0], facial_landmark[10][1]], # 우측볼 x좌표 + 이마 끝 y좌표
                        [facial_landmark[352][0], facial_landmark[152][1]], # 우측볼 x좌표 + 턱 끝 y좌표
                        [facial_landmark[123][0], facial_landmark[152][1]]] # 좌측볼 x좌표 + 턱 끝 y좌표
        warp_img = warping(cut_img, facial_landmark, WIDTH_WRAPING, HEIGHT_WRAPING, warp_point)

    else:
        warp_img = cut_img


    if outputType == 0:
        # cv2.imshow('img', img)
        # cv2.imshow('polygon', mask)
        cv2.imshow('test', cut_img)
        # cv2.imshow('warp', warp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif outputType == 1:

        #cv2.imwrite(os.path.join('./crop dataset', 'mask_'+area +'_'+ imgName), masked_cut_img)
        #cv2.imwrite(os.path.join('./crop dataset', area +'_'+ imgName), cut_img)
        cv2.imwrite(os.path.join('./part dataset', area +'_'+ imgName), warp_img)
        return
        
    elif outputType == 2:
        return cut_img


if __name__=="__main__":
    data_dir1 = './infinic dataset/normal'
    data_dir2= './face dataset'
    imgName1 = 'kyo_face1.jpg'
    imgName2 = 'N_221.png'
    
    parts = ['right_cheek','left_cheek','right_eye','left_eye','forehead']

    for (root, directories, files) in os.walk(data_dir1):
        for file in files:
            for part in parts:
                cropImage(root, file, part, 1)
            
    


    # create_landmark_img(data_dir1, imgName2, 1)
    # cropImage(data_dir2, 'black.jpg', 'all', 1)
    
    # for part in parts:
    #     cropImage(data_dir, imgName, part, 1)
    
