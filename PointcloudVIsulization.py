# import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import glob


PATH = '../../dettrain_20220711'
Lidar_Path = ''
Label_Path = ''


# match files label
def PCVisualization(lidarPath):
    pointcloud = np.fromfile(lidarPath, dtype=np.float32)
    pointcloud = pointcloud.reshape((-1, 4))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs = pointcloud[:, 0][::20]
    ys = pointcloud[:, 1][::20]
    zs = pointcloud[:, 2][::20]

    ax.scatter(xs, ys, zs)


def ReadLabelInOneFile(labelPath):
    with open(labelPath) as json_file:
        data = json.load(json_file)
        raw = {}
        boundingBoxes = data['bounding_boxes']
        for box in boundingBoxes:
            for k, v in box.items():
                if k == 'center':
                    for kk, vv in v.items():
                        if kk in raw:
                            raw[kk].append(vv)
                        else:
                            raw[kk] = [vv]
                else:
                    if k in raw:
                        raw[k].append(v)
                    else:
                        raw[k] = [v]
        print(raw)


def GetAllTrainFile():
    lidar_files = []
    label_files = []
    # for date in os.listdir(PATH):
    for date in  ['south_gate_1680_8Feb2022','south_gate_2Dec2020']:
        # Get lidar file
        # print(os.path.join(PATH, date, 'Data'))
        print(os.listdir(os.path.join(PATH, date)))
        print(len(sum([files for path, subdir, files in os.walk(os.path.join(PATH, date, 'Data')) if not subdir],[])))
                                   
        
        lidar_files.extend(sorted([file
                                   for path, subdir, files in os.walk(os.path.join(PATH, date, 'Data'))
                                   for file in glob.glob(os.path.join(path, "*.bin"))]))

        # label_files.extend(sorted([file
        #                            for path, subdir, files in os.walk(os.path.join(PATH, date, 'Label'))
        #                            for file in glob(os.path.join(pgitath, "*bin.json"))]))

    # print(len(lidar_files),len(label_files))
    print(len(lidar_files))

    print('-----------------------------------------------------------------')
    # checking
    # for i in lidar_files:
    #     for j in label_files:
    #         if i.split("\\")[-1] == j.split("\\")[-1]:
    #             continue
    #         else:
    #             print(i, j)
    #             break

    # return [lidar_files, label_files]


if __name__ == "__main__":
    # lidar_files, label_files = GetAllTrainFile()
    GetAllTrainFile()
    # PCVisualization(lidar_files[0])
    # ReadLabelInOneFile(label_files[0])
