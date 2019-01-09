# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:44:25 2018

@author: @adriantoto
"""

#Libraries
import cv2
import numpy as np

#Classifier
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#input
videoCam = cv2.VideoCapture(0)

#recognizer
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.yml")

#id varible declaration
id = 0

#font 
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while(True):
    cond, frame = videoCam.read()
    
   #gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
   #face detection using classifier
    muka = face.detectMultiScale(gray, 1.3, 5)
   
   #rectangle
    for(x, y, w, h) in muka:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        
        #id mapping--------------------------------------------------
        #put your ID and name in here
        
        if id == 1 : name = "Adrian"
        elif id == 2 : name = "Elon Musk"
        elif id == 3 : name = "George Hotz"
        
        #------------------------------------------------------------
        
        cv2.putText(frame, name, (x, y+h), font, 2, (0, 255, 0), 2)
   
   #output          
    cv2.imshow("Face Identification", frame)
    if((cv2.waitKey(1) & 0xff) == ord('q')):
        break

#after break
videoCam.release()
