import open3d
import open3d.visualization.gui as gui

import matplotlib
import numpy as np
import pandas as pd
import os
from config import OutPutVehecleClasees, Parameters
import matplotlib.path as mpltPath


CutPath = r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\tune_lidar\non-valid'
PolygonPath = [os.path.join(CutPath, x) for x in os.listdir(CutPath)]
Valid = r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\tune_lidar\valid_polygon1.csv"

box_colormap = [
    [0.2, 0.2, 0.5],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0.9, 0.1, 0.3],
    [0, 1, 1],
    [1, 0, 1],
    [0.1, 0.5, 0.4],
    [0.2, 0.7, 0.1],
    [0.6, 0.2, 0.8],
    [0.4, 0.6, 0.2],
    [0.7, 0.2, 0.5],
    [0.3, 0.8, 0.1],
    [0.8, 0.6, 0.2],
    [0.2, 0.5, 0.9],
]


def CountNumPoint(vertices, pointpath):
    polygon = mpltPath.Path(vertices)
    points = np.fromfile(pointpath, dtype=np.float32).reshape((-1, 4))
    print(points.shape)
    inside = polygon.contains_points(points[:, :2])
    print(np.count_nonzero(inside))


def GetPolygon(path):
    df = pd.read_csv(path)
    polygon = mpltPath.Path(df.iloc[:, :2].values.tolist())
    return polygon

# Get Polygon


def GetInsidePolygon(points, validPolygon, NonValidPolygonlist):
    inside = validPolygon.contains_points(points[:, :2])
    points = np.hstack((points, np.reshape(inside, (-1, 1))))
    points = points[[points[:, -1] == True]][:, :4]
    for i in NonValidPolygonlist:
        outside = i.contains_points(points[:, :2])
        points = np.hstack((points, np.reshape(outside, (-1, 1))))
        points = points[[points[:, -1] == False]][:, :4]

    return np.array(points)


def GetTransformMatrix():
    pts = open3d.geometry.PointCloud()
    T = np.eye(4)
    T[:3, :3] = pts.get_rotation_matrix_from_xyz(
        (0.0705718, -0.2612746, -0.017035))
    T[2, 3] = 5.7
    print(T)
    return T


def Trandformation(Path, T):
    points = np.fromfile(Path, dtype=np.float32).reshape((-1, 7))
    intensity = np.reshape(points[:, 3], (-1, 1))

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    pts.transform(T)

    # R = pts.get_rotation_matrix_from_xyz((0.0705718, -0.2612746, -0.017035))
    # pts = pts.rotate(R, center=(0, 0, 0))
    # pts.translate((0, 0, 5.7))  # relative should be equal to true

    points = np.hstack((np.array(pts.points), intensity))
    return points


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
    colors_rgba = [matplotlib.colors.to_rgba_array(
        color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(PointPath, transform=False, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None):
    object_list = []

    T = GetTransformMatrix()

    app = gui.Application.instance
    app.initialize()
    # vis = open3d.visualization.Visualizer()
    vis = open3d.visualization.O3DVisualizer("Open3D", 1024, 768)
    vis.show_settings = True
    # vis.create_window()

    # vis.get_render_option().point_size = 1.0
    # vis.get_render_option().background_color = np.zeros(3)

    # draw origin

    # axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.5, origin=[0, 0, 0])
    # vis.add_geometry('axis',axis_pcd)
    # vis.update_renderer()
    # object_list.append(axis_pcd)

    # get points
    if transform:
        points = Trandformation(PointPath, T)
        validPolygon = GetPolygon(Valid)
        NonValidPolygonlist = [GetPolygon(x) for x in PolygonPath]
        points = GetInsidePolygon(
            points, validPolygon=validPolygon, NonValidPolygonlist=NonValidPolygonlist)
        print(points.shape)
        # points = GetInsidePolygon(points)
    else:
        points = np.fromfile(PointPath, dtype=np.float32).reshape((-1, 4))

    # print(points.shape)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry('Points', pts)
    draw_boundary(vis)
    # vis.update_renderer()
    object_list.append(pts)

    if gt_boxes is not None:
        vis = draw_box(vis, object_list, gt_boxes)
        # vis.update_renderer()
    if ref_boxes is not None:
        print(ref_boxes,'------------------------------')
        vis = draw_box(vis, object_list, ref_boxes,
                       (0, 1, 0), ref_labels=ref_labels, gtlist=gt_boxes[:,-1], score=ref_boxes[:,-2])
        # vis.update_renderer()

    # open3d.visualization.draw_geometries([pts,axis_pcd], window_name='Open3D2')
    # open3d.io.write_image("screenshot.png", window_name="Open3D2")

    # image = vis.capture_screen_float_buffer(False)
    # open3d.io.write_image("image.png", image)

    # vis.update_renderer()
    # vis.run()
    # vis.destroy_window()

    app.add_window(vis)
    app.run()


def draw_boundary(vis, params=Parameters()):
    # create LineSet geometry
    boundary = open3d.geometry.LineSet()
    boundary.points = open3d.utility.Vector3dVector(
        np.array([[params.x_min, params.y_min, 0], [params.x_min, params.y_max, 0], [
                 params.x_max, params.y_max, 0], [params.x_max, params.y_min, 0], [params.x_min, params.y_min, 0]])
    )
    boundary.lines = open3d.utility.Vector2iVector(
        np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    )
    boundary.colors = open3d.utility.Vector3dVector(np.array([[0, 1, 0]]))
    vis.add_geometry('boundary', boundary)

    # create the grid
    grid_points = []
    grid_lines = []
    for i in np.arange(params.x_min, params.x_max + params.x_step, params.x_step*5):
        grid_points.append([i, params.y_min, 0])
        grid_points.append([i, params.y_max, 0])
        grid_lines.append([len(grid_points)-2, len(grid_points)-1])
    for i in np.arange(params.y_min, params.y_max + params.x_step, params.x_step*5):
        grid_points.append([params.x_min, i, 0])
        grid_points.append([params.x_max, i, 0])
        grid_lines.append([len(grid_points)-2, len(grid_points)-1])
    grid_colors = [[0.5, 0.5, 0.5] for i in range(len(grid_lines))]

    grid = open3d.geometry.LineSet()
    grid.points = open3d.utility.Vector3dVector(grid_points)
    grid.lines = open3d.utility.Vector2iVector(grid_lines)
    grid.colors = open3d.utility.Vector3dVector(grid_colors)

    # add the grid to the visualizer
    vis.add_geometry('grid',grid)


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
    axis_angles = np.array([0, 0, float(gt_boxes[6]) - 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, np.eye(3), lwh)
    # print(list(box3d.get_box_points()),'-----------------------------------')
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # rotate from center alpha --> same boxbox
    line_set.rotate(rot, center=tuple(center))

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)
    return line_set, box3d


def draw_box(vis, object_list, boxes, color=(0.8, 0.8, 0.8), ref_labels=[], gtlist=[], score=[]):

    for i in range(boxes.shape[0]):

        # if len(ref_labels) :
        #     if ref_labels[i] not in gtlist:
        #         continue    

        line_set, box3d = translate_boxes_to_open3d_instance(boxes[i])
        corners = box3d.get_box_points()
        if len(ref_labels) == 0:
            line_set.paint_uniform_color(color)
        else:
            if len(ref_labels):
                color = box_colormap[int(
                    list(OutPutVehecleClasees.values()).index(ref_labels[i]))]
                line_set.paint_uniform_color(color)
            else:
                color = []

            print(ref_labels[i], '======================')

            vis.add_3d_label(np.reshape(
                corners[3], (-1, 1)), ref_labels[i])
        id = 'RefBoundingBox' + \
            str(i) if len(ref_labels) > 0 else 'GTBoundingBox'+str(i)
        vis.add_geometry(id, line_set)
        object_list.append(line_set)
        object_list.append(box3d)

        if len(score):
            vis.add_3d_label(np.reshape(
                corners[5], (-1, 1)), '%.2f' % float(score[i]))

    return vis
