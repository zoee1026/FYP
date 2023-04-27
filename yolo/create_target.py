import numpy as np
import pandas as pd
from typing import List
import math
import pickle

from readers import DataReader, Label3D
from config import Parameters

from point_pillars import iouraw

class BBox:
    def __init__(self, box: Label3D):
        self.classification = box.classification
        self.x = box.centroid[0]
        self.y=box.centroid[1]
        self.z=box.centroid[2]
        self.l=box.dimension[0]
        self.w=box.dimension[1]
        self.h=box.dimension[2]
        self.yaw=box.yaw

    def get_label(self):
        return np.array([self.x,self.y,self.z,self.l,self.w,self.h,self.yaw],dtype=np.float32)
    
class AnchorBBox:
    def __init__(self, box: List):
        # ,l,w,h,z,yaw
        self.x = 0
        self.y=0
        self.z=box[3]
        self.l=box[0]
        self.w=box[1]
        self.h=box[2]
        self.yaw=0
        self.diag=math.sqrt(math.pow(self.l,2)+math.pow(self.w,2))

    def get_label(self):
        return [self.x,self.y,self.z,self.l,self.w,self.h,self.yaw]
    
    def set_xyyaw(self,x,y,yaw):
        self.x=x
        self.y=y
        self.yaw=yaw

def createTarget(labels: List[Label3D]):

    knn=pickle.load(open('knn.sav', 'rb'))

    params = Parameters()
    xSize = (params.x_max - params.x_min) / (params.x_step * params.downscaling_factor)
    ySize = (params.y_max - params.y_min) / (params.y_step * params.downscaling_factor)
    nbAnchors=params.anchor_dims.shape[0]
    nbObjects=len(labels)

    target=np.zeros((nbObjects,int(xSize),int(ySize),int(nbAnchors),10))

    posCnt = 0
    negCnt = 0
    occu=0
    objectCount = 0

    if (nbAnchors <= 0):print("Anchor length is zero")
    if (nbObjects <= 0):print("Object length is zero")

    AnchorBoxList= [AnchorBBox(i) for i in params.anchor_dims]

    for label in labels:
        
        labelBox=BBox(label)
        search_diag=math.sqrt(math.pow(labelBox.l,2)+math.pow(labelBox.w,2))/2
        
        # Calculate x_offset and y_offset
        x_offset = math.ceil(search_diag / (params.x_step * params.downscaling_factor))
        y_offset = math.ceil(search_diag / (params.y_step * params.downscaling_factor))

        # Calculate xC and yC
        xC = math.floor((labelBox.x - params.x_min) / (params.x_step * params.downscaling_factor))
        yC = math.floor((labelBox.y - params.y_min) / (params.y_step * params.downscaling_factor))

        # Calculate xStart and yStart
        xStart = max(xC - x_offset, 0)
        yStart = max(yC - y_offset, 0)

        # Calculate xEnd and yEnd
        xEnd = min(xC + x_offset, xSize)
        yEnd = min(yC + y_offset, ySize)

        maxIou = 0
        bestAnchor=None
        bestAnchorId = 0
        bestAnchor_xId = 0
        bestAnchor_yId = 0

        for xId in range(xStart, xEnd):
            x = xId * params.x_step * params.downscaling_factor + params.x_min
            # Iterate through every box within search diameter in y axis
            for yId in range(yStart, yEnd):
                # Get the real world y coordinates
                y = yId * params.x_step * params.downscaling_factor + params.y_min
                anchorCount = 0

                # For every anchor box (4 in our case)
                # Note that we are checking every anchor box for every label in the file
                for anchorBox in AnchorBoxList:
                    anchorBox.set_xyyaw(x,y,knn.predict(np.array([[x,y]]))[0])

                    iouOverlap = iouraw(anchorBox.get_label(), labelBox.get_label()) # Get IOU between two 3D boxes.

                    if maxIou < iouOverlap:
                        bestAnchor = anchorBox
                        bestAnchorId = anchorCount
                        bestAnchor_xId = xId
                        bestAnchor_yId = yId
                        maxIou = iouOverlap

                    if iouOverlap > params.positive_iou_threshold: # Accept the Anchor. Add the anchor details to the target.
                        # Tensor at CurrentObject Id, xth grid cell, yth grid cell, currentAnchor, 0
                        target[objectCount, xId, yId, anchorCount, 0] = 1

                        diag = anchorBox.diag
                        target[objectCount, xId, yId, anchorCount, 1] = (labelBox.x - anchorBox.x) / diag # delta x,y,z
                        target[objectCount, xId, yId, anchorCount, 2] = (labelBox.y - anchorBox.y) / diag
                        target[objectCount, xId, yId, anchorCount, 3] = (labelBox.z - anchorBox.z) / anchorBox.height

                        target[objectCount, xId, yId, anchorCount, 4] = math.log(labelBox.l / anchorBox.l) # delta l,w,h
                        target[objectCount, xId, yId, anchorCount, 5] = math.log(labelBox.w / anchorBox.w)
                        target[objectCount, xId, yId, anchorCount, 6] = math.log(labelBox.h / anchorBox.h)

                        target[objectCount, xId, yId, anchorCount, 7] = (labelBox.yaw - anchorBox.yaw) # delta yaw
                        if -0.5 * math.pi < labelBox.yaw <= 0.5 * math.pi:
                            target[objectCount, xId, yId, anchorCount, 8] = 1
                        else:
                            target[objectCount, xId, yId, anchorCount, 8] = 0

                        target[objectCount, xId, yId, anchorCount, 9] = params.classes[labelBox.classification]
                        occu += 1

                    elif iouOverlap < params.negative_iou_threshold:
                        target[objectCount, xId, yId, anchorCount, 0] = 0
                    else:
                        target[objectCount, xId, yId, anchorCount, 0] = -1

                    anchorCount += 1

        if maxIou < params.positive_iou_threshold:  # Comparing maxIOU for that object obtained after checking with every anchor box
            # If none of the anchors passed the threshold, then we place the best anchor details for that object.
            negCnt += 1
            xId = bestAnchor_xId
            yId = bestAnchor_yId
            diag = np.sqrt(np.power(bestAnchor.width, 2) + np.power(bestAnchor.length, 2))

            target[objectCount, xId, yId, bestAnchorId, 0] = 1
            target[objectCount, xId, yId, bestAnchorId, 1] = (labelBox.x - bestAnchor.x) / diag
            target[objectCount, xId, yId, bestAnchorId, 2] = (labelBox.y - bestAnchor.y) / diag
            target[objectCount, xId, yId, bestAnchorId, 3] = (labelBox.z - bestAnchor.z) / bestAnchor.h
            target[objectCount, xId, yId, bestAnchorId, 4] = np.log(labelBox.l / bestAnchor.l)
            target[objectCount, xId, yId, bestAnchorId, 5] = np.log(labelBox.w / bestAnchor.w)
            target[objectCount, xId, yId, bestAnchorId, 6] = np.log(labelBox.h / bestAnchor.h)
            target[objectCount, xId, yId, bestAnchorId, 7] = (labelBox.yaw - bestAnchor.yaw)

            if (-0.5 * np.pi) < labelBox.yaw <= (0.5 * np.pi):
                target[objectCount, xId, yId, bestAnchorId, 8] = 1
            else:
                target[objectCount, xId, yId, bestAnchorId, 8] = 0

            # Class id is the classification label (0,1,2,3)
            target[objectCount, xId, yId, bestAnchorId, 9] = params.classes[labelBox.classification]
            occu += 1
        else:
            posCnt += 1

        objectCount += 1
    
    return [target.astype(np.float32),posCnt,negCnt]

