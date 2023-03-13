import open3d as o3d
import matplotlib
import numpy as np

from read_file_location import ReadFileFromPath
from open3dvis import draw_scenes

if __name__ == '__main__':
    lidar_files, label_files = ReadFileFromPath('test.csv')
    print(lidar_files[0])
    # draw_scenes(lidar_files[0])
    o3d.visualization.webrtc_server.enable_webrtc()
    cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube_red.compute_vertex_normals()
    cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    o3d.visualization.draw(cube_red)

# from xvfbwrapper import Xvfb
# vdisplay = Xvfb(width=1920, height=1080)
# vdisplay.start()

# from mayavi import mlab
# mlab.init_notebook('x3d', 800, 450)
# s = mlab.test_plot3d()
# s