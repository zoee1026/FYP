import abc
from typing import List
import json
import numpy as np
import pandas as pd
from config import VehicaleClasses


class Label3D:
    def __init__(self, classification: str, centroid: np.ndarray, dimension: np.ndarray, yaw: float):
        self.classification = classification
        self.centroid = centroid
        self.dimension = dimension
        self.yaw = yaw

    def __str__(self):
        return "GT | Cls: %s, x: %f, y: %f, z: %f,l: %f, w: %f,h: %f, yaw: %f" % (
            self.classification, self.centroid[0], self.centroid[1],self.centroid[2], self.dimension[0], self.dimension[1],self.dimension[2], self.yaw)


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
        return np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))

    @staticmethod
    def read_label(file_path: str):

        with open(file_path) as json_file:
            data = json.load(json_file)
            elements = []
            boundingBoxes = data['bounding_boxes']

            for box in boundingBoxes:
                box['center']['z']=+box['height']/2
                element = Label3D(
                        str(box["object_id"]),
                        np.array([box['center']['x'],box['center']['y'],box['center']['z']+box['height']/2], dtype=np.float32),
                        np.array([box['width'],box['length'],box['height']], dtype=np.float32),
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
