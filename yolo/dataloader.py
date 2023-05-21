import numpy as np
import pandas as pd
from typing import List
import math
import pickle

from readers import DataReader, Label3D
from config_multi import Parameters
from box_class_utiliti import BBox, AnchorBBox


def get_near_points(self, x, y, i, j):
    sub_x = x - i
    sub_y = y - j
    if sub_x > 0.5 and sub_y > 0.5:
        return [[0, 0], [1, 0], [0, 1]]
    elif sub_x < 0.5 and sub_y > 0.5:
        return [[0, 0], [-1, 0], [0, 1]]
    elif sub_x < 0.5 and sub_y < 0.5:
        return [[0, 0], [-1, 0], [0, -1]]
    else:
        return [[0, 0], [1, 0], [0, -1]]


def closest_anchor(target, l):
    ratios = np.abs(l / target)
    closest_index = np.argmin(ratios)
    return closest_index


def preprocess_true_boxes(labels: List[Label3D]):

    params = Parameters()

    num_layers = params.nb_layer
    input_shape = np.array([params.Xn_f, params.Yn_f], dtype='int32')
    grid_shapes = np.array(
        [input_shape // {0: 2, 1: 1}[l] for l in range(num_layers)], dtype='int32')
    print('grid shape', grid_shapes.shape)
    anchors_mask = params.anchors_mask
    print('anchor mask', anchors_mask)
    anchor = params.anchor_dims
    anchor_l = list(anchor[..., 0])
    print('anchor length', anchor_l)

    # centroid*3, loc*3 ,yaw, occu

    y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(params.anchors_mask[l]), 8 + params.nb_classes),
                       dtype='float32') for l in range(num_layers)]
    print('y_true shape', [i.shape for i in y_true])

    for label in labels:
        n = closest_anchor(label.centroid[0], anchor_l)
        best_anchor = AnchorBBox(anchor[n])
        for l in range(num_layers):
            if n not in anchors_mask[l]:
                continue

            xx = (label.centroid[0]-params.x_min)/params.x_step
            yy = (label.centroid[1]-params.y_min)/params.y_step
            ii = xx/input_shape[0] * grid_shapes[l][1]
            jj = yy/input_shape[1] * grid_shapes[l][1]
            i = int(math.floor(ii))
            j = int(math.floor(jj))
            offsets = get_near_points(ii, jj, i, j)

            for offset in offsets:
                local_i = i + offset[0]
                local_j = j + offset[1]

                if local_i >= grid_shapes[l][1] or local_i < 0 or local_j >= grid_shapes[l][0] or local_j < 0:
                    continue

                # yaw_i=(local_i/grid_shapes[l][0]*input_shape[0])
                # yaw_j=(local_j/grid_shapes[l][1]*input_shape[0])

                # x=yaw_i*params.x_step+params.x_min
                # y=yaw_j*params.y_step+params.y_min

                y_true[l][local_j, local_i, n, 0] = 1

                y_true[l][local_j, local_i, n, 1] = label.centroid[0]
                y_true[l][local_j, local_i, n, 2] = label.centroid[1]
                y_true[l][local_j, local_i, n, 3] = label.centroid[2]
                y_true[l][local_j, local_i, n, 4] = label.dimension[0]
                y_true[l][local_j, local_i, n, 5] = label.dimension[1]
                y_true[l][local_j, local_i, n, 6] = label.dimension[2]
                y_true[l][local_j, local_i, n, 7] = label.yaw
                # y_true[l][local_j, local_i, n, 1] = (label.centroid[0]-x)/best_anchor.diag
                # y_true[l][local_j, local_i, n, 2] = (label.centroid[1]-y)/best_anchor.diag
                # y_true[l][local_j, local_i, n, 3] = (label.centroid[2]-best_anchor.z)/best_anchor.h
                # y_true[l][local_j, local_i, n, 4] = math.log(label.dimension[0]-best_anchor.l)
                # y_true[l][local_j, local_i, n, 5] = math.log(label.dimension[1]-best_anchor.w)
                # y_true[l][local_j, local_i, n, 6] = math.log(label.dimension[2]-best_anchor.h)
                # y_true[l][local_j, local_i, n, 7] = label.yaw-mapp[yaw_i,yaw_j]
                y_true[l][local_j, local_i, n, 8+label.classification] = 1

    print(y_true.shape)
    return y_true
