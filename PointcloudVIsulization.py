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

class Label3D:
    def __init__(self, classification: str, centroid: np.ndarray, dimension: np.ndarray, yaw: float):
        self.classification = classification
        self.centroid = centroid
        self.dimension = dimension
        self.yaw = yaw

    def __str__(self):
        return "GT | Cls: %s, x: %f, y: %f, l: %f, w: %f, yaw: %f" % (
            self.classification, self.centroid[0], self.centroid[1], self.dimension[0], self.dimension[1], self.yaw)


# match files label
def PCVisualization(lidarPath):
    # pointcloud = np.fromfile(lidarPath, dtype=np.float32)
    # pointcloud = pointcloud.reshape((-1, 4))
    # print(pointcloud.shape)

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # xs = pointcloud[:, 0]#[::20]
    # ys = pointcloud[:, 1]#[::20]
    # zs = pointcloud[:, 2]#[::20]

    # ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    # ax.scatter(xs, ys, zs, s=0.01)
    # ax.grid(False)
    # ax.axis('off')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    # ax.view_init(elev=40, azim=180)
    plt.plot([1,2,3], [1,2,3])
    plt.show()

def ReadLabelInOneFile(labelPath):
    # with open(labelPath) as json_file:
    #     data = json.load(json_file)
    #     raw = {}
    #     boundingBoxes = data['bounding_boxes']
    #     for box in boundingBoxes:
    #         for k, v in box.items():
    #             if k == 'center':
    #                 for kk, vv in v.items():
    #                     if kk in raw:
    #                         raw[kk].append(vv)
    #                     else:
    #                         raw[kk] = [vv]
    #             else:
    #                 if k in raw:
    #                     raw[k].append(v)
    #                 else:
    #                     raw[k] = [v]
    #     print(raw)

    with open(labelPath) as json_file:
        data = json.load(json_file)
        elements = []
        boundingBoxes = data['bounding_boxes']

        for box in boundingBoxes:
            element = Label3D(
                    str(box["object_id"]),
                    np.array(box['center'].values(), dtype=np.float32),
                    np.array([box['height'],box['width'],box['length']], dtype=np.float32),
                    float(box['angle'])
                )

            if element.classification == "DontCare":
                continue
            else:
                elements.append(element)
            print(element)  


def GetAllTrainFile():
    lidar_files = []
    label_files = []
    for date in os.listdir(PATH):
    # for date in ['south_gate_1680_8Feb2022', 'south_gate_2Dec2020']:
        # Get lidar file
        label_dir = [i for i in os.listdir(
            os.path.join(PATH, date)) if 'Label' in i][0]
        subdir=os.listdir(os.path.join(PATH, date))
            
        # print(len(sum([files for path, subdir, files in os.walk(os.path.join(PATH, date, 'Data')) if not subdir],[])))
        if "Data" in subdir:
            lidar_files.extend(sorted([file
                                    for path, subdir, files in os.walk(os.path.join(PATH, date, 'Data'))
                                    for file in glob.glob(os.path.join(path, "*.bin"))]))

        label_files.extend(sorted([file
                                   for path, subdir, files in os.walk(os.path.join(PATH, date, label_dir))
                                   for file in glob.glob(os.path.join(path, "*bin.json"))]))
        print(len(lidar_files),len(label_files))

    lidar_files, label_files=[sorted(lidar_files), sorted(label_files)]

    lidar_files_match=[]
    label_files_match=[]
    print('-----------------------------------------------------------------')
    # checking
    for i in range(len(label_files)):
        filenametarget=label_files[i].split("/")[-1].split('.')[0]
        filename=[file for file in lidar_files if filenametarget in file]
        if len(filename)==1:
            label_files_match.append(label_files[i])
            lidar_files_match.extend(filename)

    print(len(lidar_files_match),len(label_files_match))
    match_data=pd.DataFrame({"lidar_files":lidar_files_match,"label_files":label_files_match})
    match_data.to_csv('MatchFileFeb16.csv')

    return [lidar_files_match, label_files_match]

def GetMatchedDatafile(Path):
    df=pd.read_csv(Path)
    return [df['lidar_files'].tolist(),df['label_files'].tolist()]

if __name__ == "__main__":
    # lidar_files, label_files = GetAllTrainFile()

    DataPath='MatchFileFeb16.csv'
    lidar_files, label_files = GetMatchedDatafile(DataPath)
    
    # GetAllTrainFile()
    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print(lidar_files[0],label_files[0])
    PCVisualization(lidar_files[0])
    ReadLabelInOneFile(label_files[0])
# 