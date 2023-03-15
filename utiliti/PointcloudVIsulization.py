# import cv2
import datetime

# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import glob


PATH = '/media/sdb1/dettrain_20220711'
Lidar_Path = ''
Label_Path = ''

VehicaleClasses = {

    "one-box": 0,
    "three-box": 0,
    "two-box": 0,
    "black-one-box": 0,
    "black-three-box": 0,
    "black-two-box": 0,

    "taxi": 1,
    "privateminibus": 2,
    "publicminibus": 3,
    "motorbike": 4,
    "pedestrian": 5,

    "construction-vehicle": 6,
    "crane-truck": 6,
    "cylindrical-truck": 6,


    "black-cargo-mpv": 7,
    "cargo-mpv": 7,

    "black-mpv": 8,
    "mpv": 8,


    "smalltruck": 9,
    "black-smalltruck": 9,

    "black-cargo-one-box": 10,
    "cargo-one-box": 10,

    "mediumtruck": 11,
    "bigtruck": 12,
    "flatbed-truck": 13,
    "coachbus": 14,
    "dd": 15,
}
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
    # plt.plot([1,2,3], [1,2,3])
    # plt.show()
    pass

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
        return (pd.DataFrame(raw))

    # with open(labelPath) as json_file:
    #     data = json.load(json_file)
    #     elements = []
    #     boundingBoxes = data['bounding_boxes']

    #     for box in boundingBoxes:
    #         element = Label3D(
    #                 str(box["object_id"]),
    #                 np.array(list(box['center'].values()), dtype=np.float32),
    #                 np.array([box['height'],box['width'],box['length']], dtype=np.float32),
    #                 float(box['angle'])
    #             )

    #         if element.classification == "DontCare":
    #             continue
    #         else:
    #             elements.append(element)
    #         print(element)  

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
        filenametarget='/'.join(label_files[i].split("/")[-2:].split('.')[0])
        filename=[file for file in lidar_files if filenametarget in file]
        if len(filename)==1:
            label_files_match.append(label_files[i])
            lidar_files_match.extend(filename)

    print(len(lidar_files_match),len(label_files_match))
    match_data=pd.DataFrame({"lidar_files":lidar_files_match,"label_files":label_files_match})
    print(match_data['label_files'].nunique(),match_data["lidar_files"].nunique())
    match_data.to_csv('/media/sdb1/zoe/FYP/folder_root/MatchFile.csv')

    return [lidar_files_match, label_files_match]

def GetMatchedDatafile(Path):
    df=pd.read_csv(Path)
    return [df['lidar_files'].tolist(),df['label_files'].tolist()]

def GetTestClasses():
    classes=list(VehicaleClasses.keys())
    Path='/media/sdb1/zoe/FYP/folder_root/MatchFile.csv'
    lidar_files_match=[]
    label_files_match=[]
    lidar_files, label_files = GetMatchedDatafile(Path)
    print(len(lidar_files),len(label_files))

    for i in range(len(label_files)):
        with open(label_files[i]) as json_file:
            data = json.load(json_file)
            boundingBoxes = data['bounding_boxes']
            if len(boundingBoxes)==1 and boundingBoxes[0]["object_id"]=='dontcare':
                continue
            classlist=[box['object_id'] for box in boundingBoxes if box['object_id'] in classes]
            if classlist :
                lidar_files_match.append(lidar_files[i])
                label_files_match.append(label_files[i])

    print(len(lidar_files_match),len(label_files_match))
    if len(lidar_files_match)==len(label_files_match):
        match_data=pd.DataFrame({"lidar_files":lidar_files_match,"label_files":label_files_match})
        match_data.to_csv('TestFile.csv')

def ReadAllLable(path):
    df= pd.DataFrame()
    filepaths=pd.read_csv(path)['label_files'].tolist()
    for i in filepaths:
        raw=ReadLabelInOneFile(i)
        df=pd.concat([df,raw])
    print(df.info())
    df.to_csv('LabelSummary.csv')

def ReadRootFile(path):
    df=pd.read_csv(path)
    print(df.loc[0,:])

if __name__ == "__main__":
    # lidar_files, label_files = GetAllTrainFile()

    # DataPath='/media/sdb1/zoe/FYP/folder_root/MatchFile.csv'
    # DataPath=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\folder_root\MatchFile.csv'
    # df=pd.read_csv(DataPath)
    # df['Transformed']=df['lidar_files'].map(lambda x: x.split('/')[-1])
    # print(df['Transformed'].nunique(),df["lidar_files"].nunique())
    # ReadRootFile(DataPath)
    # lidar_files, label_files = GetMatchedDatafile(DataPath)
    GetAllTrainFile()
    # ReadAllLable(DataPath)
    # print(lidar_files[0],label_files[0])
    # PCVisualization(lidar_files[0])
    # ReadLabelInOneFile(label_files[0])

    # GetTestClasses()

# 

