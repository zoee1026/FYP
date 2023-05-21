import math
import numpy as np
from readers import Label3D
from typing import List


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