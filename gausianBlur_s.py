from socket import socket
import cv2 
import numpy,os,tqdm
from glob import glob
import random
import matplotlib.pylab as plt

SOURCE_IMAGES = os.path.join("/scratch","scratch1","dr_data","normal","train/")  #CHANGE THIS PATH WHEREVER REQUIRED 

def scaleRadius(img , scale):
    x=img[img.shape[0]//2,:,:].sum(1)
    r=(x > x.mean()/10 ).sum()/2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s ,fy=s)

scale=350
count = 0
for f in glob(os.path.join(SOURCE_IMAGES,"*.JPG")):

    
    a=cv2.imread(f)
    #print(a)
    
    #scale image to a given radius
    a=scaleRadius(a,scale)

    #subtract local mean color
    a=cv2.addWeighted(a, 4 ,cv2.GaussianBlur(a ,(0,0),scale/35) ,-4 ,128)


    #remove outer 10%
