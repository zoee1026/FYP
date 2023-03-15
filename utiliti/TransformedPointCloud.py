import open3d
import numpy as np
import os
import pandas as pd
import matplotlib.path as mpltPath
# from shapely.geometry import Point, Polygon

CutPath='/media/sdb1/zoe/FYP/tune_lidar/non-valid'
PolygonPath=[os.path.join(CutPath, x) for x in os.listdir(CutPath)]
Valid='/media/sdb1/zoe/FYP/tune_lidar/valid_polygon1.csv'
ToDir='/media/sdb1/zoe/FYP/train_files/'
DataPath='/media/sdb1/zoe/FYP/folder_root/TestFile.csv'
SaveCleanFile='/media/sdb1/zoe/FYP/folder_root/CleanedFiles.csv'
TransformedFiles= os.listdir(ToDir)

def GetTransformMatrix():
    pts = open3d.geometry.PointCloud()
    T = np.eye(4)
    T[:3, :3] = pts.get_rotation_matrix_from_xyz((0.0705718, -0.2612746,-0.017035))
    T[2, 3]=5.7
    return T

def GetPolygon(path):
    df=pd.read_csv(path)
    # polygon=Polygon(list(df.iloc[:,:2].to_records(index=False)))
    polygon = mpltPath.Path(df.iloc[:,:2].values.tolist())

    return polygon

# Get Polygon
def GetInsidePolygon(points,validPolygon,NonValidPolygonlist):
#     for point in points:
        # if validPolygon.contains(Point(tuple(point[:2]))):
        #     include=True
        #     for i in NonValidPolygonlist:
        #         if i.contains(Point(tuple(point[:2]))):
        #             include=False
        #             break
        #     if include:new_points.append(point)
    inside=validPolygon.contains_points(points[:, :2])
    points=np.hstack((points, np.reshape(inside,(-1,1))))
    points=points[[points[:,-1]==True]][:,:4]
    for i in NonValidPolygonlist:
        outside=i.contains_points(points[:, :2])
        points=np.hstack((points, np.reshape(outside,(-1,1))))
        points=points[[points[:,-1]==False]][:,:4]
    
    return np.array(points)

def Trandformation(Path,T):
    points = np.fromfile(Path, dtype=np.float32).reshape((-1, 7))
    points=points[:,:4]
    intensity=np.reshape(points[:, 3], (-1, 1))

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    
    # R = pts.get_rotation_matrix_from_xyz((0.0705718, -0.2612746,-0.017035))
    # pts=pts.rotate(R,center=False) # roate around coordinate center
    # pts.translate((0, 0,5.7))

    pts.transform(T)

    points=np.hstack((np.array(pts.points), intensity))
    return points

def WriteToBin(points, fileName):
    path=os.path.join(ToDir,fileName.split('/')[-1])
    points=np.reshape(points,(-1,1)).astype(np.float32)
    points.tofile(path)
    return path

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
 
    validPolygon=GetPolygon(Valid)
    NonValidPolygonlist=[GetPolygon(x) for x in PolygonPath ]

    df=pd.read_csv(DataPath)
    df['Transformed']=0

    T=GetTransformMatrix()
    print(len(TransformedFiles))

    for i in range(len(df)):
        lidar_path=df.iloc[i,0]

        if lidar_path.split('/')[-1] in TransformedFiles: continue

        points=Trandformation(lidar_path,T)
        points=GetInsidePolygon(points, validPolygon=validPolygon,NonValidPolygonlist=NonValidPolygonlist)
        print(i, points.shape,'--------------------------------------------')

        tranformed_path=WriteToBin(points, lidar_path)
        df.iloc[i,2]=tranformed_path

    df.to_csv(SaveCleanFile)
