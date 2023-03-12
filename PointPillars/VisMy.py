# import open3d
# import matplotlib
# import numpy as np

# from read_file_location import ReadFileFromPath
# from open3dvis import draw_scenes

# if __name__ == '__main__':
#     lidar_files, label_files = ReadFileFromPath('test.csv')
#     draw_scenes(lidar_files[0])

from xvfbwrapper import Xvfb
vdisplay = Xvfb(width=1920, height=1080)
vdisplay.start()

from mayavi import mlab
mlab.init_notebook('x3d', 800, 450)
s = mlab.test_plot3d()
s