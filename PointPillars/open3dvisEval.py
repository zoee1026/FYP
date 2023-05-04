import numpy as np
import os
import pandas as pd
import json
from open3dvis import draw_scenes, CountNumPoint
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
            element = list(box['center'].values())+[box['width'], box['length'], box['height']]+[
                box['angle'], OutPutVehecleClasees[VehicaleClasses[str(box["object_id"])]]]
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
            classes=box[0]
            elements.append(box[1:]+[classes])

        print(elements)
        return np.array(elements)


if __name__ == '__main__':
    resultPath=''
    # PointPath=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\train_files\2020_12_03=00_03_32_387.bin'
    # LabelPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_03=00_03_35_387.bin.json"
    PointPath = r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_lidar\2020_12_02=08_10_22_845.bin'
    LabelPath = r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_02=08_10_22_845.bin.json"
    # resultPath = r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_result\2020_12_02=08_10_22_845.txt'

    label_elements = ReadLabelInOneFile(LabelPath)
    # labels=[VehicaleClasses[x] for x in list(label_elements[:,-1])]


    if resultPath:
        result_elements = ReadResultInOneFile(resultPath)
        print(result_elements[:, -1])
        draw_scenes(PointPath=PointPath, transform=False, gt_boxes=label_elements,
                    ref_boxes=result_elements, ref_labels=result_elements[:, -1])
    else:
        draw_scenes(PointPath=PointPath,transform=False,gt_boxes=label_elements)


    # CountNumPoint(pointpath=PointPath, vertices=np.array(
    #     [[-21.25832748,  -1.55922395], [-16.85124969,  -1.55922395], [-21.25832748,   0.14472836],[-16.85124969,   0.14472836]]))
