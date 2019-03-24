import h5py
import sys,os
import keras
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import cv2
def secclastrain():
    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
        # ESC pressed
            print("Closing the program.")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            path='classif2'
            cv2.imwrite(os.path.join(path , img_name), frame)
            print("{} written!".format(img_name))
            img_counter += 1
                #    cam.release()
                #    cv2.destroyAllWindows()
    return img_counter
def firtest():
    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
        # ESC pressed
            print("Closing the program.")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            path='test1'
            cv2.imwrite(os.path.join(path , img_name), frame)
            print("{} written!".format(img_name))
            img_counter += 1
#    cam.release()
#    cv2.destroyAllWindows()
    return img_counter
def firclastrain():
    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            print("Closing the program.")
            break
        elif k%256 == 32:
            img_name = "opencv_frame_{}.png".format(img_counter)
            path='classif1'
            cv2.imwrite(os.path.join(path , img_name), frame)
            print("{} written!".format(img_name))
            img_counter += 1
#    cam.release()
#    cv2.destroyAllWindows()
    return img_counter
def sectest():
    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
        # ESC pressed
            print("Closing the program.")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            path='test2'
            cv2.imwrite(os.path.join(path , img_name), frame)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()    
    return img_counter
cam = cv2.VideoCapture(0)
rop=0
cv2.namedWindow("test")
print("Train your data \n Press what you wish to do \n 1.Train with perfect parts. \n 2.Train with defected parts \n 3.Put Test perfect images \n 4.Put test defected part/n")
rop = int(input())
img_counter=0
for i in range(0,4):
    img_counter=firclastrain()
    img_counter=secclastrain()
    img_counter=firtest()
    img_counter=sectest()


     
cam.release()

cv2.destroyAllWindows()
