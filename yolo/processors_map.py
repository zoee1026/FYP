from typing import List
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import Sequence

from config import Parameters
from point_pillars import createPillars
# from create_target import createTarget
from create_target_map import createTarget

from readers import DataReader, Label3D
from sklearn.utils import shuffle
import sys


def select_best_anchors(arr):
    # arr -> [num_GT_BB, grid_x, grid_y, num_anchors, 10]
    dims = np.indices(arr.shape[1:])
    ind = (np.argmax(arr[..., 0:1], axis=0),) + tuple(dims)
    return arr[ind]


class DataProcessor(Parameters):

    def __init__(self):
        super(DataProcessor, self).__init__()
        anchor_dims = np.array(self.anchor_dims, dtype=np.float32)
        self.anchor_dims = anchor_dims[:, 0:3]  # length, width, height
        self.anchor_z = anchor_dims[:, 3]  # z-center
        self.anchor_yaw = anchor_dims[:, 4]  # yaw-angle
        # Counts may be used to make statistic about how well the anchor boxes fit the objects
        self.pos_cnt, self.neg_cnt = 0, 0

    def make_point_pillars(self, points: np.ndarray, print_flag: bool = False):

        assert points.ndim == 2
        assert points.shape[1] == 4
        assert points.dtype == np.float32

        pillars, indices = createPillars(points,
                                         self.max_points_per_pillar,
                                         self.max_pillars,
                                         self.x_step,
                                         self.y_step,
                                         self.x_min,
                                         self.x_max,
                                         self.y_min,
                                         self.y_max,
                                         self.z_min,
                                         self.z_max,
                                         print_flag)

        return pillars, indices

    def make_ground_truth(self, labels: List[Label3D],mapp):

        # filter labels by classes (cars, pedestrians and Trams)
        # Label has 4 properties (Classification (0th index of labels file),
        # centroid coordinates, dimensions, yaw)
        labels = list(
            filter(lambda x: x.classification in self.classes, labels))

        target, pos, neg = createTarget(labels,mapp)
        self.pos_cnt += pos
        self.neg_cnt += neg

        # return a merged target view for all objects in the ground truth and get categorical labels
        sel = select_best_anchors(target)
        ohe = tf.keras.utils.to_categorical(
            sel[..., 9], num_classes=self.nb_classes, dtype='float64')

        return sel[..., 0], sel[..., 1:4], sel[..., 4:7], sel[..., 7], sel[..., 8], ohe


class SimpleDataGenerator(DataProcessor, Sequence):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, data_reader: DataReader, batch_size: int, lidar_files: List[str], y_map, label_files: List[str] = None,):
        #  calibration_files: List[str] = None):
        super(SimpleDataGenerator, self).__init__()
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.lidar_files = lidar_files
        self.label_files = label_files
        # self.model=kdt
        self.yaw_map=y_map

    def __len__(self):
        return len(self.lidar_files) // self.batch_size

    def __getitem__(self, batch_id: int):
        file_ids = np.arange(batch_id * self.batch_size,
                             self.batch_size * (batch_id + 1))
        #         print("inside getitem")
        pillars = []
        voxels = []
        occupancy = []
        position = []
        size = []
        angle = []
        heading = []
        classification = []

        for i in file_ids:

            assert self.lidar_files[i].split("/")[-1].split('.')[0] == self.label_files[i].split("/")[-1].split('.')[0]

            lidar = self.data_reader.read_lidar(self.lidar_files[i])
            # For each file, dividing the space into a x-y grid to create pillars
            # Voxels are the pillar ids
            pillars_, voxels_ = self.make_point_pillars(lidar)

            pillars.append(pillars_)
            voxels.append(voxels_)

            if self.label_files is not None:
                label = self.data_reader.read_label(self.label_files[i])
                occupancy_, position_, size_, angle_, heading_, classification_ = self.make_ground_truth(
                    label, self.yaw_map)
                occupancy.append(occupancy_)
                position.append(position_)
                size.append(size_)
                angle.append(angle_)
                heading.append(heading_)
                classification.append(classification_)

        pillars = np.concatenate(pillars, axis=0)
        voxels = np.concatenate(voxels, axis=0)


        if self.label_files is not None:
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            return [pillars, voxels], [occupancy, position, size, angle, heading, classification]
        else:
            return [pillars, voxels]

    def on_epoch_end(self):
        #         print("inside epoch")
        if self.label_files is not None:
            self.lidar_files, self.label_files = \
                shuffle(self.lidar_files, self.label_files,)
