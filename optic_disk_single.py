"""
Date: 19/07/2018
Task: To find mark to position of optic disc

"""
import cv2
import numpy

def optic_disk_extraction(image):
	# the area of the image with the largest intensity value
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
	(a,b)= maxLoc
	vertex1 = (a-60,b+60)
   	vertex2 = (a+60,b-60)	
	cv2.rectangle(image, vertex1, vertex2, 2)
	#cv2.imshow("marker",image)
	return image






if __name__ == '__main__':
	fundus1=cv2.imread("original.JPG")
	fundus1=cv2.resize(fundus1, (512,512))
	optic_disk1 = optic_disk_extraction(fundus1)
	cv2.imwrite("optic_disk1.jpeg",optic_disk1)
cv2.waitKey(0)
