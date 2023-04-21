# import math
# from random import shuffle, sample

# import cv2
# import keras
# import numpy as np
# from PIL import Image

# def preprocess_true_boxes(
#                 objectPositions,
#                 objectDimensions,
#                 objectYaws,
#                 objectClassIds,
#                 anchorDimensions,
#                 anchorZHeights,
#                 anchorYaws,
#                 positiveThreshold,
#                 negativeThreshold,
#                 nbClasses,
#                 downscalingFactor,
#                 xStep,
#                 yStep,
#                 xMin,
#                 xMax,
#                 yMin,
#                 yMax,
#                 zMin,
#                 zMax,
#         true_boxes, input_shape, anchors, num_classes=16):

#         scaled_mask=[2,4,8]

#         def input_shape(downscalingFactor):
#             return [math.floor((xMax - xMin) / (xStep * downscalingFactor)),math.floor((yMax - yMin) / (yStep * downscalingFactor))]
        
#         grid_shapes = np.array([input_shape(i) for i in scaled_mask], dtype='int32')

#         nbAnchors = anchorDimensions.shape()[0]
#         nbObjects = objectDimensions.shape()[0]
        
#         #-----------------------------------------------------------#
#         #   一共有三个特征层数
#         #-----------------------------------------------------------#
#         num_layers  = len(self.anchors_mask)
#         #-----------------------------------------------------------#
#         #   m为图片数量，grid_shapes为网格的shape
#         #   20, 20  640/32 = 20
#         #   40, 40
#         #   80, 80
#         #-----------------------------------------------------------#
#         m           = true_boxes.shape[0]
#         grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
#         #-----------------------------------------------------------#
#         #   y_true的格式为
#         #   m,20,20,3,5+num_classses
#         #   m,40,40,3,5+num_classses
#         #   m,80,80,3,5+num_classses
#         #-----------------------------------------------------------#
#         y_true = [np.zeros((nbObjects, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes),
#                     dtype='float32') for l in range(num_layers)]
#         #-----------------------------------------------------#
#         #   用于帮助先验框找到最对应的真实框
#         #-----------------------------------------------------#
#         box_best_ratios = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l])),
#                     dtype='float32') for l in range(num_layers)]

#         #-----------------------------------------------------------#
#         #   通过计算获得真实框的中心和宽高
#         #   中心点(m,n,2) 宽高(m,n,2)
#         #-----------------------------------------------------------#
#         boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
#         boxes_wh =  true_boxes[..., 2:4] - true_boxes[..., 0:2]
#         #-----------------------------------------------------------#
#         #   将真实框归一化到小数形式
#         #-----------------------------------------------------------#
#         true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
#         true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

#         #-----------------------------------------------------------#
#         #   [9,2] -> [9,2]
#         #-----------------------------------------------------------#
#         anchors         = np.array(anchors, np.float32)

#         #-----------------------------------------------------------#
#         #   长宽要大于0才有效
#         #-----------------------------------------------------------#
#         valid_mask = boxes_wh[..., 0]>0

#         for b in range(m):
#             #-----------------------------------------------------------#
#             #   对每一张图进行处理
#             #-----------------------------------------------------------#
#             wh = boxes_wh[b, valid_mask[b]]

#             if len(wh) == 0: continue
#             #-------------------------------------------------------#
#             #   wh                          : num_true_box, 2
#             #   np.expand_dims(wh, 1)       : num_true_box, 1, 2
#             #   anchors                     : 9, 2
#             #   np.expand_dims(anchors, 0)  : 1, 9, 2
#             #   
#             #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
#             #   ratios_of_gt_anchors    : num_true_box, 9, 2
#             #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的宽高的比值
#             #   ratios_of_anchors_gt    : num_true_box, 9, 2
#             #
#             #   ratios                  : num_true_box, 9, 4
#             #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
#             #   max_ratios              : num_true_box, 9
#             #-------------------------------------------------------#
#             ratios_of_gt_anchors = np.expand_dims(wh, 1) / np.expand_dims(anchors, 0)
#             ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(wh, 1)
#             ratios               = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)
#             max_ratios           = np.max(ratios, axis = -1)
            
#             for t, ratio in enumerate(max_ratios):
#                 #-------------------------------------------------------#
#                 #   ratio : 9
#                 #-------------------------------------------------------#
#                 over_threshold = ratio < self.threshold
#                 over_threshold[np.argmin(ratio)] = True
#                 #-----------------------------------------------------------#
#                 #   找到每个真实框所属的特征层
#                 #-----------------------------------------------------------#
#                 for l in range(num_layers):
#                     for k, n in enumerate(self.anchors_mask[l]):
#                         if not over_threshold[n]:
#                             continue
#                         #-----------------------------------------------------------#
#                         #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
#                         #-----------------------------------------------------------#
#                         i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
#                         j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')
#                         offsets = self.get_near_points(true_boxes[b,t,0] * grid_shapes[l][1], true_boxes[b,t,1] * grid_shapes[l][0], i, j)
#                         for offset in offsets:
#                             local_i = i + offset[0]
#                             local_j = j + offset[1]

#                             if local_i >= grid_shapes[l][1] or local_i < 0 or local_j >= grid_shapes[l][0] or local_j < 0:
#                                 continue

#                             if box_best_ratios[l][b, local_j, local_i, k] != 0:
#                                 if box_best_ratios[l][b, local_j, local_i, k] > ratio[n]:
#                                     y_true[l][b, local_j, local_i, k, :] = 0
#                                 else:
#                                     continue
#                             #-----------------------------------------------------------#
#                             #   c指的是当前这个真实框的种类
#                             #-----------------------------------------------------------#
#                             c = true_boxes[b, t, 4].astype('int32')
#                             #-----------------------------------------------------------#
#                             #   y_true的shape为(m,20,20,3,85)(m,40,40,3,85)(m,80,80,3,85)
#                             #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
#                             #   1代表的是置信度、80代表的是种类
#                             #-----------------------------------------------------------#
#                             y_true[l][b, local_j, local_i, k, 0:4] = true_boxes[b, t, 0:4]
#                             y_true[l][b, local_j, local_i, k, 4] = 1
#                             y_true[l][b, local_j, local_i, k, 5+c] = 1
#                             box_best_ratios[l][b, local_j, local_i, k] = ratio[n]

#         return y_true
