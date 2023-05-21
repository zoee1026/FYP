import tensorflow as tf
import numpy as np
from keras import backend as K
from readers import DataReader, Label3D
from config_multi import Parameters
from box_class_utiliti import BBox, AnchorBBox
import cv2 as cv


def get_anchors_and_decode(feats, anchors, num_classes, input_shape, mapp, scale, calc_loss=False):

    params = Parameters()
    num_anchors = len(anchors)
    grid_shape = K.shape(feats)[:2]
    scale = np.int32(scale)

    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [
                    1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = K.tile(K.reshape(K.arange(
        0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))

    anchors_tensor = K.cast(K.tile(
        anchors.reshape(1, 1, *anchors.shape), [grid_shape[0], grid_shape[1], 1, 1]), K.dtype(feats))

    # def cal_dig(row):
    #     return np.sqrt(np.sum([np.power(row[0],2),np.power(row[1],2)]))
    # anchors_diag = K.cast(np.apply_along_axis(cal_dig,1,anchors_tensor),K.dtype(feats))
    feats = K.reshape(
        feats, [grid_shape[0], grid_shape[1], num_anchors, num_classes + 8])

    map_tensor = mapp[::scale, ::scale]
    map_tensor = K.cast(K.tile(map_tensor.reshape(
        [grid.shape[0], grid.shape[1], 1, 1]), (1, 1, num_anchors, 1)), K.dtype(feats))

    box_x = K.expand_dims((feats[..., 1]*anchors_tensor[..., -1]+grid[..., 0]) *
                          K.constant(params.x_step)*scale+K.constant(params.x_min), axis=-1)

    box_y = K.expand_dims((feats[..., 2]*anchors_tensor[..., -1]+grid[..., 1]) *
                          K.constant(params.y_step)*scale+K.constant(params.y_min), axis=-1)

    box_z = K.expand_dims(
        feats[..., 3]*anchors[..., 2]+anchors_tensor[..., 3], axis=-1)
    box_yaw = K.expand_dims(feats[..., 7]+map_tensor[..., 0], axis=-1)
    box_l = K.expand_dims(K.exp(feats[..., 4]*anchors_tensor[..., 0]), axis=-1)
    box_w = K.expand_dims(K.exp(feats[..., 5]*anchors_tensor[..., 1]), axis=-1)
    box_h = K.expand_dims(K.exp(feats[..., 6]*anchors_tensor[..., 2]), axis=-1)

    # ------------------------------------------#
    box_confidence = K.expand_dims(K.sigmoid(feats[..., 0]), axis=-1)
    box_class_probs =K.sigmoid(feats[..., 8:])

    boxes = K.concatenate([box_confidence, box_x, box_y, box_z,
                          box_l, box_w, box_h, box_yaw,  box_class_probs], axis=-1)

    if calc_loss == True:
        return grid, boxes, box_confidence, feats
    return boxes, box_confidence, box_class_probs


def DecodeBox(outputs, mapp,
              confidence=0.5,
              max_boxes=10,
              nms_iou=0.3):

    params = Parameters()
    num_layers = params.nb_layer
    anchor_mask = params.anchors_mask
    anchor = params.anchor_dims
    input_shape = np.array([params.Xn_f, params.Yn_f], dtype='float32')
    num_classes = params.nb_classes

    boxes = []
    box_confidence = []
    box_class_probs = []

    for i in range(len(outputs)):
        box, box_conf, sub_box_class_probs = get_anchors_and_decode(
            outputs[i], anchor[anchor_mask[i]], num_classes, input_shape, mapp, params.scale[i])

        boxes.append(K.reshape(box, [-1, 8+num_classes]))
        box_confidence.append(K.reshape(box_conf, [-1, 1]))
        box_class_probs.append(
            K.reshape(sub_box_class_probs, [-1, num_classes]))
    boxes = K.concatenate(boxes, axis=0)
    print('total box', boxes.shape)
    box_confidence = K.concatenate(box_confidence, axis=0)
    box_class_probs = K.concatenate(box_class_probs, axis=0)
    box_scores = box_confidence * box_class_probs

    # -----------------------------------------------------------#
    #   判断得分是否大于score_threshold
    # -----------------------------------------------------------#
    mask = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out = []
    scores_out = []
    classes_out = []
    # -----------------------------------------------------------#
    #   筛选出一定区域内属于同一种类得分最大的框
    # -----------------------------------------------------------#
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_bbox = class_boxes[:, [1, 2, 4, 5, 7]].numpy().tolist()
        class_bbox[:, -1] = np.rad2deg(class_bbox[:, -1])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        indices = cv.dnn.NMSBoxesRotated(
            class_bbox, class_box_scores, confidence, nms_iou)

        # nms_index = tf.image.non_max_suppression(
        #     class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)
        output = class_boxes[indices]
        boxes_out.append(output)
    print(boxes_out.shape, '---------------------')
    return boxes_out
