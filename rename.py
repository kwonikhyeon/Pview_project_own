import sys
import os
import cv2
import numpy as np

def rename(fileDir, oldName):
    #파일명 받아서 spilt (-기준) 해서 나오는 값중에 숫자
    #1->73 ... 27->99 ... 28->1 ... 99-> 72
    #조건문으로 28보다 작으면 +72해준다.
    #28 이상이면 -27한다.
    #함수 밖에서 만약 카운트랑 번호랑 다르면..?
    #검정이미지 새로 생성하고 넘긴다. 560*560
    #근데 u-net input size는 560임

    nameList = oldName.split('-')
    fileNum = int(nameList[1])
    if fileNum < 28:
        fileNum += 72
    else:
        fileNum -= 27

    newName = 'label' + '_' + '{0:03d}'.format(fileNum) + '.png'
    file_oldname = os.path.join(fileDir, oldName)
    file_newname_newfile = os.path.join(fileDir, newName)

    os.rename(file_oldname, file_newname_newfile)

    return newName

def resizeLabel(fileDir, imgName, checkList):
    #라벨 저장경로
    save_dir = './wrinkle dataset/label'
    #라벨 resize 크기
    size = 560
    
    img = cv2.imread(os.path.join(data_dir, imgName))
    img = cv2.resize(img, (size,size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #기존 라벨 번호 추출
    nameList = imgName.replace('.', '_').split('_')
    fileNum = int(nameList[1])

    #빈 라벨 추가
    while True:
        if len(checkList) == 0 or (checkList[-1] + 1) == fileNum:
            break
        emptyLabel = np.zeros((size, size), dtype=np.uint8)
        emptyLabelNum = checkList[-1] + 1
        emptyLabelName = 'label_' + '{0:03d}'.format(emptyLabelNum) + '.png'
        cv2.imwrite(os.path.join(data_dir, emptyLabelName), emptyLabel)
        checkList.append(emptyLabelNum)
        print(emptyLabelName, ' is appended!')

    #크기 조정 이미지 저장
    cv2.imwrite(os.path.join(data_dir, imgName), img)
    cv2.imwrite(os.path.join(save_dir, imgName), img)
    checkList.append(fileNum)
    checkList.sort()

    return


if __name__ == '__main__':
    
    data_dir = './crop label'
    check = []
    exist = []

    for (root, directories, files) in os.walk(data_dir):
        for file in files:
            # result = rename(root, file)
            # if result not in check:
            #     check.append(result)
            # else:
            #     exist.append(result)
            resizeLabel(root, file, check)


