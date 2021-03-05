import cv2
import os
import numpy as np

#Conv_Filter kernel size 3x3
KERNAL_SIZE = 3
STRIDE = 1
PADDING =  (KERNAL_SIZE - STRIDE)/2
PADDING = int(PADDING)
print(PADDING)
img = cv2.imread('./006_01_01_051_08.png')
img = cv2.resize(img,(28,32))
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#Conv_Filter Pooling Std=0, mean=2, kernel size 3x3
Conv_Filter = np.random.normal(2,0,(3,3))
Conv_Filter = Conv_Filter/np.sum(Conv_Filter)
img_F = img

if (PADDING % 1) == 0:
    img_F = np.pad(img_F,((PADDING,PADDING),(PADDING,PADDING)),'constant')
else:
    print(PADDING)
    img_F = np.pad(img_F,((PADDING*2,PADDING*2),(0,0)),'constant')
[H,W]=np.shape(img_F)

new_feature = np.zeros((int((H-KERNAL_SIZE)/STRIDE)+1,int((W-KERNAL_SIZE)/STRIDE)+1))

for h in range(int((H-KERNAL_SIZE)/STRIDE)+1):
    for w in range(int((W-KERNAL_SIZE)/STRIDE)+1):
        aa = img_F[h*STRIDE:h*STRIDE + (KERNAL_SIZE), w*STRIDE:w*STRIDE + (KERNAL_SIZE)]*Conv_Filter

        new_feature[h,w] = np.sum(aa)

        img_S = img_F.astype(np.uint8)
        img_new = new_feature.astype(np.uint8)

        cv2.rectangle(img_S, (int(w*STRIDE), int(h*STRIDE)), (int((w*STRIDE + KERNAL_SIZE)), int((h*STRIDE + KERNAL_SIZE))), (255, 0, 0), 1)
        cv2.namedWindow('Conv_process', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Conv_process", 300, 300)
        cv2.imshow('Conv_process', img_S)

        cv2.namedWindow('Conv_result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Conv_result", 300, 300)
        cv2.imshow('Conv_result', img_new)

        cv2.waitKey(20)
[H,W]=np.shape(img_new)
img_P = np.zeros((int(H/2),int(W/2)))
stride_x = 0
stride_y = 0
for h in range(int(H/2)):
    for w in range(int(W/2)):
        step_y = h+stride_y
        step_x = w+stride_x
        bb = img_new[int(step_y):int(step_y+2),int(step_x):int(step_x+2)]
        print(bb)

        img_P[h,w] = np.max(bb)
        stride_x = stride_x +1
    stride_x = 0
    stride_y = stride_y +1
img_P = img_P.astype((np.uint8))

cv2.namedWindow('MaxPooling', cv2.WINDOW_NORMAL)
cv2.resizeWindow("MaxPooling", 300, 300)
cv2.imshow('MaxPooling', img_P)

cv2.waitKey(0)

print('The shape of input before max pooling is %dx%d after is %dx%d\n\n' %(img_new.shape[0],img_new.shape[1],img_P.shape[0],img_P.shape[1]))



