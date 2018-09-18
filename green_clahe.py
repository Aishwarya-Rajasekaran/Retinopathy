
# Apply CLAHE on Green Channel image
import cv2
import numpy as np
import os
from glob import glob
count=0
#source : change path when required
SOURCE_IMAGES = os.path.join("/home","meera","Desktop","normal/") 

for f in glob(os.path.join(SOURCE_IMAGES,"*.jpg")):

	img= cv2.imread(f)
	img= cv2.resize(img, (512,512))
	green_channel = img[:, :, 1]

	#Apply Clahe twice
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cla = clahe.apply(green_channel)
	result_img = clahe.apply(cla)
	#cv2.imshow('clahe 2',cla)

	path = "/home/meera/Desktop/normal_result"  #CHANGE PATH ACCORDINGLY
	
	cv2.imwrite(os.path.join(path,'clahe_{}.jpg'.format(count)),result_img)  
	count += 1    

cv2.waitKey(0)
print("done")
print(count)


