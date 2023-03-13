import abc
from typing import List
import json
import numpy as np
import pandas as pd
from config import VehicaleClasses
import open3d


class Label3D:
    def __init__(self, classification: str, centroid: np.ndarray, dimension: np.ndarray, yaw: float):
        self.classification = classification
        self.centroid = centroid
        self.dimension = dimension
        self.yaw = yaw

    def __str__(self):
        return "GT | Cls: %s, x: %f, y: %f, l: %f, w: %f, yaw: %f" % (
            self.classification, self.centroid[0], self.centroid[1], self.dimension[0], self.dimension[1], self.yaw)


class DataReader:

    @staticmethod
    @abc.abstractmethod
    def read_lidar(file_path: str) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_label(file_path: str) -> List[Label3D]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_calibration(file_path: str) -> np.ndarray:
        raise NotImplementedError


class KittiDataReader(DataReader):

    def __init__(self):
        super(KittiDataReader, self).__init__()

    @staticmethod
    def read_lidar(file_path: str):
        points=np.fromfile(file_path, dtype=np.float32).reshape((-1, 7))
        intensity=np.reshape(points[:, 3], (-1, 1))

        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        
        pts.translate((0, 0,5.7))
        R = pts.get_rotation_matrix_from_xyz((0.0705718, -0.2612746,-0.017035))
        pts=pts.rotate(R, center=(0,0,0))
        
        points=np.hstack((np.array(pts.points), intensity))
        return points

    @staticmethod
    def read_label(file_path: str):

        with open(file_path) as json_file:
            data = json.load(json_file)
            elements = []
            boundingBoxes = data['bounding_boxes']

            for box in boundingBoxes:
                element = Label3D(
                        str(box["object_id"]),
                        np.array(list(box['center'].values()), dtype=np.float32),
                        np.array([box['length'],box['width'],box['height']], dtype=np.float32),
                        float(box['angle'])
                    )
                # if element.classification =="dontcare":
                if element.classification not in list(VehicaleClasses.keys()):
                    continue
                else:
                    # print (element)
                    elements.append(element)

            return elements

    @staticmethod
    def read_calibration(file_path: str):
        with open(file_path, "r") as f:
            lines = f.readlines()
            Tr_velo_to_cam = np.array(lines[5].split(": ")[1].split(" "), dtype=np.float32).reshape((3, 4))
            R, t = Tr_velo_to_cam[:, :3], Tr_velo_to_cam[:, 3]
            return R, t
