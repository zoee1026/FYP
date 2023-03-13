
import open3d
import numpy as np
import os
import pandas as pd
from shapely.geometry import Point, Polygon
import struct

CutPath='/media/sdb1/zoe/FYP/Tune/non-valid'
PolygonPath=[os.path.join(CutPath, x) for x in os.listdir(CutPath)]
Valid='/media/sdb1/zoe/FYP/Tune/valid_polygon1.csv'

# CutPath=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\Tune\non-valid'
# PolygonPath=[os.path.join(CutPath, x) for x in os.listdir(CutPath)]
# Valid=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\Tune\valid_polygon1.csv"

ToDir='/media/sdb1/zoe/FYP/train_files/'

def GetPolygon(path):
    df=pd.read_csv(path)
    polygon=Polygon(list(df.iloc[:,:2].to_records(index=False)))
    return polygon

# Get Polygon
def GetInsidePolygon(points,validPolygon,NonValidPolygonlist):
    new_points=[]
    for point in points:
        if validPolygon.contains(Point(tuple(point[:2]))):
            include=True
            for i in NonValidPolygonlist:
                if i.contains(Point(tuple(point[:2]))):
                    include=False
                    break
            if include:new_points.append(point)
    
    return np.array(new_points)

def Trandformation(Path):
    points = np.fromfile(Path, dtype=np.float32).reshape((-1, 7))
    intensity=np.reshape(points[:, 3], (-1, 1))

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    
    R = pts.get_rotation_matrix_from_xyz((0.0705718, -0.2612746,-0.017035))
    pts=pts.rotate(R)
    pts.translate((0, 0,5.7))

    points=np.hstack((np.array(pts.points), intensity))
    return points

def WriteToBin(points, fileName):
    path=os.path.join(ToDir,fileName.split('/')[-1])
    # path=fileName.split('\\')[-1]
    print(path)
    points=np.reshape(points,(-1,1)).astype(np.float32)
    points.tofile(path)
    return path
    # with open(path, 'wb') as f:
    #     # Write number of points as 4-byte integer
    #     f.write(struct.pack('<I', points.shape[0]))
    #     # Write points to file as 32-bit floats
    #     f.write(points.astype(np.float32).tobytes())

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    validPolygon=GetPolygon(Valid)
    NonValidPolygonlist=[GetPolygon(x) for x in PolygonPath ]

    DataPath='TestFile.csv'
    df=pd.read_csv(DataPath)
    df['Transformed']=0
    print(df.shape)

    for i in range(len(df)):
        lidar_path=df.iloc[i,0]
        print(lidar_path)
        print('label',df.iloc[i,1])

        points=Trandformation(lidar_path)
        points=GetInsidePolygon(points, validPolygon=validPolygon,NonValidPolygonlist=NonValidPolygonlist)
        print(points.shape,'----------------------------------')

        tranformed_path=WriteToBin(points, lidar_path)
        df.iloc[i,2]=tranformed_path
        print(df.shape,'===============================================')

        break

    # lidar_path=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_lidar\2020_12_03=08_37_03_798.bin"
    # LabelPath=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_03=08_37_03_798.bin.json"
    # points=Trandformation(lidar_path)
    # points=GetInsidePolygon(points, validPolygon=validPolygon,NonValidPolygonlist=NonValidPolygonlist)
    # print(points.shape,'============================================')
    # tranformed_path=WriteToBin(points, lidar_path)
   
    df.to_csv('CleanedFiles.csv')


       

    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
