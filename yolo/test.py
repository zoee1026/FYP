import math
import numpy as np
import pandas as pd
from dataloader import preprocess_true_boxes
from readers import KittiDataReader, DataReader, Label3D
from config import Parameters


def make_ground_truth(labels):

    params = Parameters()

    # filter labels by classes (cars, pedestrians and Trams)
    # Label has 4 properties (Classification (0th index of labels file),
    # centroid coordinates, dimensions, yaw)
    labels = list(
        filter(lambda x: x.classification in params.classes, labels))
    print([label.classification for label in labels])

    # For each label file, generate these properties except for the Don't care class
    target_positions = np.array(
        [label.centroid for label in labels], dtype=np.float32)
    target_dimension = np.array(
        [label.dimension for label in labels], dtype=np.float32)
    target_yaw = np.array(
        [label.yaw for label in labels], dtype=np.float32)

    # change from str to int representing classes label
    target_class = np.array([params.classes[label.classification]
                            for label in labels], dtype=np.int32)

    # assert np.all(target_yaw >= -np.pi) & np.all(target_yaw <= np.pi)
    assert len(target_positions) == len(
        target_dimension) == len(target_yaw) == len(target_class)

    ytrue = preprocess_true_boxes(
        target_positions,
        target_dimension,
        target_yaw,
        target_class,
        params.anchor_dims[:, 0:3],
        params.anchor_dims[:, 3],
        params.anchor_dims[:, 4],
        params.positive_iou_threshold,
        params.negative_iou_threshold,
        params.x_step,
        params.y_step,
        params.x_min,
        params.x_max,
        params.y_min,
        params.y_max,
        params.z_min,
        params.z_max,
        params.anchors_mask,
        num_classes=16,
    )

    return ytrue


if __name__ == '__main__':
    label_path = r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_03=00_03_35_387.bin.json'
    reader=KittiDataReader()
    labels = reader.read_label(label_path)
    make_ground_truth(labels)
