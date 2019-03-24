import h5py
import sys,os
import keras
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
    # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        #img = "opencv_frame_{}.png".format(img_counter)
        #cv2.imwrite(img, frame)
        # print("{} written!".format(img))
        tup=(127,127)
        imgr = cv2.resize(frame, tup, interpolation = cv2.INTER_AREA)
        #print(frame)
        
        #imgr=im.resize(tup)
        img_counter += 1
        pred=[]
        pred.append(imgr)
        pre=np.asarray(pred)
        model=load_model('arth.h5')
        p=model.predict(pre)
        print(p)
        if(p[0][0]==0):
            print("it's perfect")
            janstr='This is a perfect'
        else:
            print("it's defected")
            janstr='This is imperfect'
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 3
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(frame,janstr,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
        cv2.imshow("Result of the input",frame)

cam.release()

cv2.destroyAllWindows()


