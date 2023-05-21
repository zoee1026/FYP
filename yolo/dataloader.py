import numpy as np
import pandas as pd
from typing import List
import math
import pickle

from readers import DataReader, Label3D
from config_multi import Parameters
from box_class_utiliti import BBox, AnchorBBox


def get_near_points(x, y, i, j):
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
    input_shape = np.array([params.Xn, params.Yn], dtype='int32')
    grid_shapes = np.array(
        [input_shape // {0: 2, 1: 1}[l] for l in range(num_layers)], dtype='int32')
    anchors_mask = params.anchors_mask
    anchor = params.anchor_dims
    anchor_l = list(anchor[..., 0])

    # centroid*3, loc*3 ,yaw, occu

    y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(params.anchors_mask[l]), 8 + params.nb_classes),
                       dtype='float32') for l in range(num_layers)]

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
            index=n//3

            for offset in offsets:
                local_i = i + offset[0]
                local_j = j + offset[1]

                if local_i >= grid_shapes[l][1] or local_i < 0 or local_j >= grid_shapes[l][0] or local_j < 0:
                    continue

                y_true[l][local_j, local_i, index, 0] = 1

                y_true[l][local_j, local_i, index, 1] = label.centroid[0]
                y_true[l][local_j, local_i, index, 2] = label.centroid[1]
                y_true[l][local_j, local_i, index, 3] = label.centroid[2]
                y_true[l][local_j, local_i, index, 4] = label.dimension[0]
                y_true[l][local_j, local_i, index, 5] = label.dimension[1]
                y_true[l][local_j, local_i, index, 6] = label.dimension[2]
                y_true[l][local_j, local_i, index, 7] = label.yaw
        
                y_true[l][local_j, local_i, n, 8+params.classes[label.classification]] = 1

    return y_true
