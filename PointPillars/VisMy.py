import open3d 
import matplotlib
import numpy as np

from read_file_location import ReadFileFromPath
from open3dvis import draw_scenes

if __name__ == '__main__':
    Path=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\test.csv"
    lidar_files, label_files = ReadFileFromPath(Path)
    # print (lidar_files[:10])
    # print (label_files[:10])
    # print(lidar_files[0])
    # draw_scenes(lidar_files[0])
    draw_scenes(r"C:\Users\Chan Kin Yan\Desktop\FYP\KittiData\testing\velodyne\000000.bin")

