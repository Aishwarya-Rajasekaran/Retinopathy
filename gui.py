#input_image=tkFileDialog.askopenfilename()
import Tkinter as tk 
import cv2
from PIL import ImageTk,Image
import numpy as np
from matplotlib import pyplot as plt
import tkFileDialog

global img

def upload():
	file = tkFileDialog.askopenfilename()
	img = ImageTk.PhotoImage(Image.open(file))
	l1.configure(image=img)
	l1.image = img
	print "inside upload"


r=tk.Tk()
r.title('Retinopathy')
width=r.winfo_screenwidth()
height=r.winfo_screenheight()
print width,height
r.geometry(("%dx%d")%(width,height))

title_label=tk.Label(r,text="BLOOD VESSEL EXTRACTION",fg="red",font="Verdana 20 bold",pady=15)
title_label.grid(row=1,column=3)

l1=tk.Label(r,height=512,width=512)
l1.grid(row=2,column=2) 


b1 = tk.Button(r,text='Upload image', width= 25,command= upload())
b1.grid(row=3,column=2) 	

"""
b2 = tk.Button(r,text='Extract Blood Vessels', width= 25)
b2.grid(row=3,column=2) 	
b2.bind('<Button>',extract_blood_vessel(inp))
"""
r.mainloop()

#cv2.waitKey(0)