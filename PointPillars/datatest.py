import pandas as pd
from read_file_location import GetMatchedDatafile
import glob
import os

# Path='../MatchFile.csv'
trainPath= "/media/sdb1/kitti/object/training"
lidar_files = sorted(glob.glob(os.path.join(trainPath, "velodyne", "*.bin")))
label_files = sorted(glob.glob(os.path.join(trainPath, "label_2", "*.txt")))
calibration_files = sorted(glob.glob(os.path.join(trainPath, "calib", "*.txt")))
# df = pd.read_csv(Path)
print (len(label_files),len(lidar_files), len(calibration_files))


