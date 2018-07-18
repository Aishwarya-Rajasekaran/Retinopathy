""""
Date  : 17/07/2018 
Task  : Blood Vessel Extraction from fundus image using morphological operation

Note: 
1. Dont use adaptive thresholding technique as there are many globs.
2. foreground should be white while doing any operation
3. Matplotlib colour format is BGR!
4. image= cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot(titles,images):
	for i in xrange(6):
    		plt.subplot(2,3,i+1)
    		plt.imshow(images[i],'gray')
    		plt.title(titles[i])
    	plt.show()
	

def extract_blood_vessel(image):
	green_channel=image[:,:,1]
	clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	contrast_img =clahe.apply(green_channel)
	contrast_img =clahe.apply(contrast_img)
	     
    # To remove major noise
	o1 = cv2.morphologyEx(contrast_img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
	c1 = cv2.morphologyEx(o1,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
	o2 = cv2.morphologyEx(c1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations = 1)
	c2 = cv2.morphologyEx(o2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations = 1)
	o3 = cv2.morphologyEx(c2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations = 1)
	c3 = cv2.morphologyEx(o3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations = 1)

	f1 = cv2.subtract(c3,contrast_img)
	f2 = clahe.apply(f1)
   	ret,f3 = cv2.threshold(f2,15,255,cv2.THRESH_BINARY)

    # To remove minor noise
 	mask = np.ones(f2.shape[:2], dtype="uint8") * 255	
	im, contours, hierarchy = cv2.findContours(f3.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) <= 150:
			cv2.drawContours(mask, [cnt], -1, 0, -1)   
	f4 = cv2.bitwise_and(f2, f2, mask=mask)
    blood_vessel=f4;	

 	titles = ['Green Channel','Contrast Enhancement','Morphological Operations','Subtraction','Binary','Noise Removal']
	images = [green_channel,contrast_img,c3,f2,f3,blood_vessel]
	plot(titles,images)

	return blood_vessel


if __name__ == '__main__':
	fundus=cv2.imread('original.JPG')
	fundus=cv2.resize(fundus,(512,512))
	blood_vessel = extract_blood_vessel(fundus)
	cv2.imwrite("bv.JPG", blood_vessel)
cv2.waitKey(0)



"""

	# To remove big blobs like exudates,cotton wool.(PRESENTLY NOT REQUIRED )
 	ret,fin = cv2.threshold(f4,15,255,cv2.THRESH_BINARY_INV)	
	fundus_eroded = cv2.bitwise_not(fin)	

	xmask = np.ones(fundus.shape[:2], dtype="uint8") * 255
	x1, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
	for cnt in xcontours:
		shape = "unknown"
		perimeter = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, False)   				
		if len(approx) > 4 and cv2.contourArea(cnt) <= 1000 and cv2.contourArea(cnt) >= 300:
			shape = "blob"	
		else:
			shape = "vessel"
		if(shape=="blob"):
			cv2.drawContours(xmask, [cnt], -1, 0, -1)	
	
	blood_vessel = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
"""