# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:44:56 2020

@author: ANIKETH
"""
from packages import social_distancing_config as config
from packages.detection import detect_people
from scipy.spatial import distance as dist 
import numpy as np 
import cv2 as cv
import os
import imutils 
labelsPath =os.path.sep.join([config.MODEL_PATH,"coco.names"])
LABELS =open(labelsPath).read().strip().split("\n")

weightsPath =os.path.sep.join([config.MODEL_PATH,"yolov3.weights"])

configPath= os.path.sep.join([config.MODEL_PATH,"yolov3.cfg"])
print("Loading YOLO from disk")
#convolution network code
net =cv.dnn.readNetFromDarknet(configPath,weightsPath)

if config.USE_GPU:
	net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    
ln =net.getLayerNames()
ln = [ln[i[0]-1]for i in net.getUnconnectedOutLayers()]
print("Accesing video stream")

vs =cv.VideoCapture( r"input2.mp4"if "pedestrian.mp4"else 0)
writer =None

while True:
    (grabbed,frame) =vs.read()
    if not grabbed:
        break
    frame = imutils.resize(frame,width =1500)
    results =detect_people(frame,net,ln,personIdx=LABELS.index("person"))
    violate =set()
    if len(results) >=2:
        
        # extract all centroids from the results and compute the Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids,centroids,metric="euclidean")
        
        
   # loop over the upper triangular of the distance matrix
   
        for i in range(0,D.shape[0]):
            for j in range(i+1,D.shape[1]):
                if D[i,j]<config.MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)
                    
                    
    for(i,(prob,bbox,centroid)) in enumerate(results):
        (startX,startY,endX,endY)=bbox
        (cX,cY)=centroid 
        color=(0,255,0)
        
        if i in violate:
            color = (0,0,255)
        cv.rectangle(frame,(startX,startY),(endX,endY),color,2)
        cv.circle(frame,(cX,cY),5,color,1)
    
    text ="Social Distancing Violations: {}".format(len(violate))
    cv.putText(frame,text,(10,frame.shape[0]-25),cv.FONT_HERSHEY_SIMPLEX,0.85,(0,0,255),3)
    cv.imshow("Frame",frame)
    if cv.waitKey(0) & 0xFF==ord('q'):
        break

    if r"social-distance-detector" != "" and writer is None:
        fourcc =cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(r"output.mp4",fourcc,25,(frame.shape[1],frame.shape[0]),True)
        
    if writer is not None:
        writer.write(frame)
        cv.destroyAllWindows()                            