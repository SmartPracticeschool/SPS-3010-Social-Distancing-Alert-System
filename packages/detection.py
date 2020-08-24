# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:19:31 2020

@author: ANIKETH
"""

from .social_distancing_config import MIN_CONFID 
from .social_distancing_config import NMS_THRESHLD

import cv2 as cv
import numpy as np

def detect_people(frame, net, ln, personIdx=0):
    

    (H, W) = frame.shape[:2]
    results = []
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personIdx and confidence > MIN_CONFID:
                centerX=int(detection[0]*W)
                centerY=int(detection[1]*H)
                width=int(detection[2]*W)
                height=int(detection[3]*H)
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    #Non Maxima Supression
    idxs = cv.dnn.NMSBoxes(boxes, confidences, MIN_CONFID, NMS_THRESHLD)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
        
#return the list of results
    return results