import open3d
import matplotlib
import numpy as np

from read_file_location import ReadFileFromPath
from open3dvis import draw_scenes

if __name__ == '__main__':
    lidar_files, label_files = ReadFileFromPath('test.csv')
    draw_scenes(lidar_files[0])