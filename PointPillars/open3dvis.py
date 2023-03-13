import open3d
import matplotlib
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
import os

# DIR=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\Tune\non-valid'

# PolygonPath=os.listdir(DIR)
# Valid=r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\Tune\valid_polygon1.csv"

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

# def GetPolygon(path):
#     df=pd.read_csv(path)
#     polygon=Polygon(list(df.iloc[:,:2].to_records(index=False)))
#     return polygon

# # Get Polygon
# def GetInsidePolygon(points):
#     new_points=[]
    
#     validPolygon=GetPolygon(Valid)
#     polygon_list=[GetPolygon(os.path.join(DIR, x)) for x in PolygonPath ]

#     for point in points:
#         if validPolygon.contains(Point(tuple(point[:2]))):
#             include=True
#             for i in polygon_list:
#                 if i.contains(Point(tuple(point[:2]))):
#                     include=False
#                     break
#             if include:new_points.append(point)
    
#     return np.array(new_points)

# def Trandformation(Path):
#     points = np.fromfile(Path, dtype=np.float32).reshape((-1, 7))
#     intensity=np.reshape(points[:, 3], (-1, 1))

#     pts = open3d.geometry.PointCloud()
#     pts.points = open3d.utility.Vector3dVector(points[:, :3])
    
#     R = pts.get_rotation_matrix_from_xyz((0.0705718, -0.2612746,-0.017035))
#     pts=pts.rotate(R, center=(0,0,0))
#     pts.translate((0, 0,5.7))

#     points=np.hstack((np.array(pts.points), intensity))
#     return points

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster
    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba



def draw_scenes(PointPath, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    object_list=[]

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin

    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    vis.update_renderer()
    object_list.append(axis_pcd)

    # get points
    # points=Trandformation(PointPath)
    # points=GetInsidePolygon(points)
    points = np.fromfile(PointPath, dtype=np.float32).reshape((-1, 4))

    print(points.shape)


    # infile = open(PointPath, "rb")
    # buf = infile.read()
    # infile.close()

    # points = np.frombuffer(buf, dtype=np.float32).reshape(-1, 7)
    # intensity=np.reshape(points[:, 3], (-1, 1))

    # pts = open3d.geometry.PointCloud()
    # pts.points = open3d.utility.Vector3dVector(points[:, :3])
    # # pts.color=open3d.utility.Vector3dVector(np.reshape(points[:, 3], (-1, 1)))

    
    # pts.translate((0, 0,5.7))
    # R = pts.get_rotation_matrix_from_xyz((0.0705718, -0.2612746,-0.017035))
    # pts=pts.rotate(R, center=(0,0,0))

    # points=np.hstack((np.array(pts.points), intensity))

    # axis = open3d.visualization.create_axes()
    # vis.add_geometry(axis)
    # vis.update_renderer()
    
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    vis.update_renderer()
    object_list.append(pts)

    if gt_boxes is not None:
        vis = draw_box(vis, object_list, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, object_list, ref_boxes, (0, 1, 0), ref_labels, ref_scores)


    # open3d.visualization.draw_geometries([pts,axis_pcd], window_name='Open3D2')
    # open3d.io.write_image("screenshot.png", window_name="Open3D2")

    vis.update_renderer()
    vis.run()

    image = vis.capture_screen_float_buffer(False)
    open3d.io.write_image("image.png", image)

    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, float(gt_boxes[6]) + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)
    return line_set, box3d


def draw_box(vis, object_list, boxes, color=(1, 0, 0), ref_labels=None, score=None):
    for i in range(boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)
        object_list.append(line_set)
        object_list.append(box3d)


        if score is not None:
            corners = box3d.get_box_points()
            vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
