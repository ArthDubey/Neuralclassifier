import h5py
import sys,os
#import keras
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

image = cv2.imread(sys.argv[1])
im=Image.open(sys.argv[1])
print(image)
tup=(127,127)
imgr = cv2.resize(image, tup, interpolation = cv2.INTER_AREA)
        #print(frame)
        
        #imgr=im.resize(tup)
pred=[]
pred.append(imgr)
pre=np.asarray(pred)
model=load_model('arth.h5')
p=model.predict(pre)
print(p[0][0])
if(p[0][0]==0):
    print("it's good")
elif(p[0][0]==1):
    print("it's bad")

