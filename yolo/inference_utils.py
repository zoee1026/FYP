import numpy as np
import cv2 as cv
from typing import List
from config import Parameters
from readers import DataReader
from processors import DataProcessor
import tensorflow as tf
from scipy.special import softmax
from readers import Label3D
import json
from config import VehicaleClasses, OutPutVehecleClasees
import glob
import os
import pandas as pd
from create_target import get_yaw


class BBox(Parameters, tuple):
    """ bounding box tuple that can easily be accessed while being compatible to cv2 rotational rects """

    def __new__(cls, bb_x, bb_y, bb_z, bb_length, bb_width, bb_height, bb_yaw, bb_heading, bb_cls, bb_conf):
        bbx_tuple = ((float(bb_x), float(bb_y)), (float(bb_length),
                     float(bb_width)), float(np.rad2deg(bb_yaw)))
        return super(BBox, cls).__new__(cls, tuple(bbx_tuple))

    def __init__(self, bb_x, bb_y, bb_z, bb_length, bb_width, bb_height, bb_yaw, bb_heading, bb_cls, bb_conf):
        super(BBox, self).__init__()

        self.x = bb_x
        self.y = bb_y
        self.z = bb_z

        self.length = bb_length
        self.width = bb_width
        self.height = bb_height

        self.z -= self.height/2.

        # self.length -= 0.3
        # self.width -= 0.3
        # self.length = self.length * (self.x_max - self.x_min)
        # self.width = self.width * (self.y_max - self.y_min)
        # self.height = self.height * (self.z_max - self.z_min)

        self.yaw = bb_yaw
        self.heading = bb_heading
        self.cls = bb_cls
        self.conf = bb_conf
        self.class_dict = {
            0: "Car",
            1: "taxi",
            2: "privateminibus",
            3: "publicminibus",
            4: "motorbike",
            5: "pedestrian",
            6: "construction-vehicle",
            7: "cargo-mpv",
            8: "mpv",
            9: "smalltruck",
            10: "cargo-one-box",
            11: "mediumtruck",
            12: "bigtruck",
            13: "flatbed-truck",
            14:  "coachbus",
            15: "dd",
        }

    def __str__(self):
        return "BB | Cls: %s, x: %f, y: %f, z: %f, l: %f, w: %f, h: %f, yaw: %f, heading: %f, conf: %f" % (
            OutPutVehecleClasees[self.cls], self.x, self.y, self.z, self.length, self.width, self.height, self.yaw, self.heading, self.conf)


    def get_labels(self):
        # if (int(self.heading) == 0) and (self.yaw < 0):
        #     self.yaw = - self.yaw
        return [self.class_dict[self.cls],
                self.x, self.y, self.z,   self.length,  self.width, self.height, self.yaw, self.conf]

def get_formated_label(boxes: List[BBox], indices: List):
    labels = []
    for idx in indices:
        labels.append(boxes[idx].get_labels())
    return labels


def dump_predictions(predictions: List, file_path: str, file_csv:pd.DataFrame):
    """ Dumps the model predictions in txt files so that it can be used by KITTI evaluation toolkit """
    with open(file_path, 'w') as out_txt_file:
        if len(predictions):
            for bbox in predictions:
                bboxList = []
                for bbox_attribute in bbox:
                    out_txt_file.write("{} ".format(bbox_attribute))
                    bboxList.append(bbox_attribute)
                out_txt_file.write("\n")

                bboxList.append(file_path)
                bboxList.append(file_path.split('/')[-1])
                print(bboxList)
                df = pd.DataFrame([bboxList])
                file_csv = pd.concat([file_csv,df], ignore_index=True)
        return file_csv


def ReadGTLabel(labelPath):
    with open(labelPath) as json_file:
        data = json.load(json_file)
        elements = []
        boundingBoxes = data['bounding_boxes']

        for box in boundingBoxes:
            element = Label3D(
                str(box["object_id"]),
                np.array([box['center']['x'], box['center']['y'],
                         box['center']['z']], dtype=np.float32),
                np.array([box['width'], box['length'],
                         box['height']], dtype=np.float32),
                float(box['angle'])
            )
            # if element.classification =="dontcare":
            if element.classification not in list(VehicaleClasses.keys()):
                continue
            else:
                print(element)
                elements.append(element)
        return elements


def rotational_nms(set_boxes, confidences, occ_threshold=0.3, nms_iou_thr=0.5):
    """ rotational NMS
    set_boxes = size NSeqs list of size NDet lists of tuples. each tuple has the form ((pos, pos), (size, size), angle)
    confidences = size NSeqs list of lists containing NDet floats, i.e. one per detection
    """
    assert len(set_boxes) == len(
        confidences) and 0 < occ_threshold < 1 and 0 < nms_iou_thr < 1
    if not len(set_boxes):
        return []
    assert (isinstance(set_boxes[0][0][0], float) or isinstance(set_boxes[0][0][0], int)) and \
           (isinstance(confidences[0], float)
            or isinstance(confidences[0], int))
    nms_boxes = []

    # If batch_size > 1
    # for boxes, confs in zip(set_boxes, confidences):
    #     assert len(boxes) == len(confs)
    #     indices = cv.dnn.NMSBoxesRotated(boxes, confs, occ_threshold, nms_iou_thr)
    #     indices = indices.reshape(len(indices)).tolist()
    #     nms_boxes.append([boxes[i] for i in indices])

    # IF batch_size == 1
    if len(set_boxes)>1:
        indices = cv.dnn.NMSBoxesRotated(
            set_boxes, confidences, occ_threshold, nms_iou_thr)
        indices = np.array(indices).reshape(len(indices)).tolist()
    # nms_boxes.append([set_boxes[i] for i in indices])
        return indices  
    else:
        return [0]


def generate_bboxes_from_pred(occ, pos, siz, ang, hdg, clf, anchor_dims,  kdt,y_map,occ_threshold=0.5):
    """ Generating the bounding boxes based on the regression targets """

    # Get only the boxes where occupancy is greater or equal threshold.
    real_boxes = np.where(occ >= occ_threshold)
    if np.array(real_boxes).shape[1] == 0:
        real_boxes = np.where(occ == np.amax(occ))

    # Get the indices of the occupancy array
    coordinates = list(zip(real_boxes[0], real_boxes[1], real_boxes[2]))
    # Assign anchor dimensions as original bounding box coordinates which will eventually be changed
    # according to the predicted regression targets
    anchor_dims = anchor_dims
    real_anchors = np.random.rand(len(coordinates), len(anchor_dims[0]))
    for i, value in enumerate(real_boxes[2]):
        real_anchors[i, ...] = anchor_dims[value]
    # Change the anchor boxes based on regression targets, this is the inverse of the operations given in
    # createPillarTargets function (src/PointPillars.cpp)
    predicted_boxes = []
    for i, value in enumerate(coordinates):
        real_diag = np.sqrt(
            np.square(real_anchors[i][0]) + np.square(real_anchors[i][1]))
        real_x = value[0] * Parameters.x_step * \
            Parameters.downscaling_factor + Parameters.x_min
        real_y = value[1] * Parameters.y_step * \
            Parameters.downscaling_factor + Parameters.y_min
        bb_x = pos[value][0] * real_diag + real_x
        bb_y = pos[value][1] * real_diag + real_y
        bb_z = pos[value][2] * real_anchors[i][2] + real_anchors[i][3]
        # print(position[value], real_x, real_y, real_diag)
        bb_length = np.exp(siz[value][0]) * real_anchors[i][0]
        bb_width = np.exp(siz[value][1]) * real_anchors[i][1]
        bb_height = np.exp(siz[value][2]) * real_anchors[i][2]
        # bb_yaw = ang[value] + real_anchors[i][4]
        bb_yaw = ang[value] + get_yaw(kdt,y_map,np.array([[bb_x,bb_y]]))
        # bb_yaw = ang[value]
        bb_heading = np.round(hdg[value])
        # if bb_heading == 0:
        #     bb_yaw -= np.pi
        bb_cls = np.argmax(softmax(clf[value]))
        bb_conf = occ[value]
        predicted_boxes.append(BBox(bb_x, bb_y, bb_z+bb_height/2, bb_length, bb_width, bb_height,
                                    bb_yaw, bb_heading, bb_cls, bb_conf))

    return predicted_boxes


def cal_precision(boxes, gt, precisions):
    gt_classes = [VehicaleClasses[i.classification] for i in gt]
    pred_classes = [i.cls for i in boxes]

    classes = set(gt_classes + pred_classes)

    for c in classes:
        TP = 0
        FP = 0
        for i, pred_c in enumerate(pred_classes):
            if pred_c == c:
                if pred_c in gt_classes:
                    TP += 1
                    precisions['TP'] += 1
                else:
                    FP += 1
                    precisions['FP'] += 1

        if TP + FP > 0:
            precision = TP / (TP + FP)
            precisions[c].append(precision)


def Get_finalPrecisions(precisions):
    overall = precisions['TP']/(precisions['TP']+precisions['FP'])
    print('Overall precision is ', overall)

    for k, v in precisions.items():
        if isinstance(k, int):
            if len(v) > 0:
                print("Precision of ",
                      OutPutVehecleClasees[k], 'is ', sum(v)/len(v))
            else:
                continue
        else:
            continue


@tf.function
def pillar_net_predict_server(inputs, model):
    """ tf.function wrapper for faster inference """
    return model(inputs, training=False)
