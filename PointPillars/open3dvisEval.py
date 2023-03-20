import numpy as np
import os
import pandas as pd
import json
from open3dvis import draw_scenes
from config import VehicaleClasses, OutPutVehecleClasees

def ReadLabelInOneFile(labelPath):
    #  [center, lwh, yaw,className]

    with open(labelPath) as json_file:
        data = json.load(json_file)
        elements = []
        boundingBoxes = data['bounding_boxes']

        for box in boundingBoxes:
            if box["object_id"] not in list(VehicaleClasses.keys()):
                continue
            element =list(box['center'].values())+[box['width'],box['length'],box['height']]+[str(box['angle']), str(box["object_id"])]
            elements.append(element)

        print(elements)  
        return np.array(elements)
    
def ReadResultInOneFile(labelPath):
    with open(labelPath, "r") as f:
        lines = f.readlines()
        elements = []
        for line in lines:
            box = line.split()
            if box[0] not in list(OutPutVehecleClasees.values()):
                continue
            elements.append(box)

        print(elements)  
        return np.array(elements)

if __name__ == '__main__':
    # PointPath=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\train_files\2020_12_03=00_03_35_387.bin'
    # LabelPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_03=00_03_35_387.bin.json"
    PointPath=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_lidar\2020_12_02=09_28_20_428.bin'
    LabelPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_02=09_28_20_428.bin.json"
    resultPath=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_result\2020_12_02=09_28_20_428.txt'
    label_elements=ReadLabelInOneFile(LabelPath)
    labels=[VehicaleClasses[x] for x in list(label_elements[:,-1])]
    result_elements=ReadResultInOneFile(label_elements)

    draw_scenes(PointPath=PointPath,transform=True,gt_boxes=label_elements,ref_boxes=result_elements, ref_labels=result_elements[:,0])

