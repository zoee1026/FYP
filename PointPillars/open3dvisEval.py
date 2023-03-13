import numpy as np
import os
import pandas as pd
import json
import glob

from open3dvis import draw_scenes
from config import VehicaleClasses

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
    LabelPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_03=08_37_05_798.bin.json"
    PointPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_lidar\2020_12_03=08_37_05_798.bin"
    # PointPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\train_files\2020_12_03=00_03_32_387.bin"
    # LabelPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_03=00_03_32_387.bin.json"
    # PointPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_lidar\2020_12_03=08_37_03_798.bin"
    # PointPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\2020_12_03=08_37_03_798.bin"
    # LabelPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_03=08_37_03_798.bin.json"

    draw_scenes(PointPath=PointPath,transform=True,gt_boxes=ReadLabelInOneFile(LabelPath))

