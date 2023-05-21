import tensorflow as tf
import numpy as np
from keras import backend as K
from readers import DataReader, Label3D
from config_multi import Parameters
from box_class_utiliti import BBox, AnchorBBox
import cv2 as cv



def get_anchors_and_decode(feats, anchors, num_classes, input_shape, mapp, calc_loss=False):

    params = Parameters()
    num_anchors = len(anchors)
    grid_shape = K.shape(feats)[:2]
    scale = int((input_shape[0]/grid_shape[0]))
    print('scale', scale)

    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [
                    1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = K.tile(K.reshape(K.arange(
        0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))

    anchors_tensor = K.tile(
        anchors.reshape(1,1,*anchors.shape), [grid_shape[0], grid_shape[1], 1, 1])
    anchors_diag = K.sqrt(K.sum(K.square(anchors_tensor[..., 0:2]), axis=1))
 
    feats = K.reshape(
        feats, [grid_shape[0], grid_shape[1], num_anchors, num_classes + 8])

    map_tensor=mapp[::scale].reshape(grid_shape)
    map_tensor = K.tile(K.expand_dims(mapp, axis=-1), [1, 1, num_anchors, 1])
    print('map_tensor shape',map_tensor.shape)

    box_x = (feats[..., 1]*anchors_diag+grid) * \
        K.constant(params.x_step)*scale+K.constant(params.x_min)
    box_y = (feats[..., 2]*anchors_diag+grid) * \
        K.constant(params.y_step)*scale+K.constant(params.y_min)
    box_z = feats[..., 3]*anchors[..., 2]+anchors_tensor[..., 3]
    box_yaw = feats[..., 7]+map_tensor
    box_l = K.exp(feats[..., 4]*anchors_tensor[..., 0])
    box_w = K.exp(feats[..., 5]*anchors_tensor[..., 1])
    box_h = K.exp(feats[..., 6]*anchors_tensor[..., 2])

    # ------------------------------------------#
    box_confidence = K.sigmoid(feats[..., 0])
    box_class_probs = K.sigmoid(feats[..., 8:])

    boxes= K.concatenate([box_confidence,box_x, box_y, box_z, box_l, box_w, box_h, box_yaw,  box_class_probs], axis=-1)
   
    print('box',boxes.shape)
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

    boxes=[]
    box_confidence = []
    box_class_probs = []

    for i in range(len(outputs)):
        box, box_conf, sub_box_class_probs= get_anchors_and_decode(
                outputs[i], anchor[anchor_mask[i]], num_classes, input_shape, mapp)
           
        boxes.append(K.reshape(box, [-1, 8+num_classes]))
        box_confidence.append(K.reshape(box_conf, [-1, 1]))
        box_class_probs.append(
            K.reshape(sub_box_class_probs, [-1, num_classes]))
    boxes = K.concatenate(boxes, axis=0)
    print('total box',boxes.shape)
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
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_bbox=class_boxes[:,[1,2,4,5,7]].numpy().tolist()
        class_bbox[:, -1] = np.rad2deg(class_bbox[:, -1])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        indices = cv.dnn.NMSBoxesRotated(
           class_bbox , class_box_scores, confidence, nms_iou)

        # nms_index = tf.image.non_max_suppression(
        #     class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)
        output=class_boxes[indices]
        boxes_out.append(output)
    print(boxes_out.shape,'---------------------')
    return boxes_out
