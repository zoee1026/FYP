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

if __name__ == '__main__':
    PointPath=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\train_files\2020_12_03=00_03_35_387.bin'
    LabelPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_03=00_03_35_387.bin.json"
    elements=ReadLabelInOneFile(LabelPath)
    labels=[VehicaleClasses[x] for x in list(elements[:,-1])]

    draw_scenes(PointPath=PointPath,transform=False,gt_boxes=elements, ref_labels=labels)

