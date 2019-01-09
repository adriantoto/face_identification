# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:01:15 2018

@author: @adriantoto
"""

#libraries
import os
import cv2
import numpy as np
from PIL import Image

#trainning tool
recognizer = cv2.face.LBPHFaceRecognizer_create()

#data set input
path = 'dataSet'

#method to get img with id
def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        #convert to gray scale
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces

#variable input to method
Ids, faces = getImagesWithID(path)

#train the training tool
recognizer.train(faces, Ids)
recognizer.save('recognizer/trainingData.yml')

cv2.destroyAllWindows()
