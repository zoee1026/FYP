from functools import wraps

import tensorflow as tf
from keras import backend as K
from keras.initializers import random_normal
from keras.layers import (Add, BatchNormalization, Concatenate, Conv2D, Layer,
                          MaxPooling2D, ZeroPadding2D)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.layers import (Concatenate, Input, Lambda, UpSampling2D,
                          ZeroPadding2D)
from keras.models import Model

from nets.CSPdarknet_4features import (C3, DarknetConv2D, DarknetConv2D_BN_SiLU,Bottleneck,
                             darknet_body,compose)
import tensorflow as tf
import numpy as np
from config import Parameters



def build_point_pillar_graph(params: Parameters, batch_size: int = Parameters.batch_size, gpu: int=1 ):

    # extract required parameters
    max_pillars = int(params.max_pillars)
    max_points  = int(params.max_points_per_pillar)
    nb_features = int(params.nb_features)
    nb_channels = int(params.nb_channels)
    base_channels=params.nb_channels
    # batch_size  = int(params.batch_size)
    image_size  = tuple([params.Xn, params.Yn])
    nb_classes  = int(params.nb_classes)
    anchors_mask =params.anchors_mask
    base_depth  = 3

    # print(image_size)

    if tf.keras.backend.image_data_format() == "channels_first":
        raise NotImplementedError
    else:
        input_shape = (max_pillars, max_points, nb_features)

    input_pillars = tf.keras.layers.Input(input_shape, batch_size=batch_size*gpu, name="pillars/input")
    input_indices = tf.keras.layers.Input((max_pillars, 3), batch_size=batch_size*gpu, name="pillars/indices",
                                          dtype=tf.int32)

    def correct_batch_indices(tensor, batch_size):
        array = np.zeros((batch_size, max_pillars, 3), dtype=np.float32)
        for i in range(batch_size):
            array[i, :, 0] = i
        return tensor + tf.constant(array, dtype=tf.int32)

    if batch_size > 1:
            corrected_indices = tf.keras.layers.Lambda(lambda t: correct_batch_indices(t, batch_size))(input_indices)
    else:
        corrected_indices = input_indices

    x=Bottleneck(input_pillars,nb_channels,shortcut=False,name="Pillars/cov")
    x = tf.keras.layers.MaxPool2D((1, max_points), name="pillars/maxpooling2d")(x)

    if tf.keras.backend.image_data_format() == "channels_first":
        reshape_shape = (nb_channels, max_pillars)
    else:
        reshape_shape = (max_pillars, nb_channels)

    x = tf.keras.layers.Reshape(reshape_shape, name="pillars/reshape")(x)
    pillars = tf.keras.layers.Lambda(lambda inp: tf.scatter_nd(inp[0], inp[1],
                                                               (batch_size,) + image_size + (nb_channels,)),
                                     name="pillars/scatter_nd")([corrected_indices, x])

    # 2d cnn backbone
    feat1, feat2, feat3,feat4 = darknet_body(pillars, base_channels, base_depth)

    P6          = DarknetConv2D_BN_SiLU(int(base_channels * 8), (1, 1), name = 'conv_for_feat4')(feat4)  
    # 20, 20, 512 -> 40, 40, 512
    P6_upsample = UpSampling2D()(P6) 
    # 40, 40, 512 cat 40, 40, 512 -> 40, 40, 1024
    P6_upsample = Concatenate(axis = -1)([P6_upsample, feat3])
    # 40, 40, 1024 -> 40, 40, 512
    P6_upsample = C3(P6_upsample, int(base_channels * 8), 1, shortcut = False, name = 'conv3_for_upsample0')
    # 20, 20, 1024 -> 20, 20, 512

    P5          = DarknetConv2D_BN_SiLU(int(base_channels * 4), (1, 1), name = 'conv_for_feat3')(P6_upsample)  
    # 20, 20, 512 -> 40, 40, 512
    P5_upsample = UpSampling2D()(P5) 
    # 40, 40, 512 cat 40, 40, 512 -> 40, 40, 1024
    P5_upsample = Concatenate(axis = -1)([P5_upsample, feat2])
    # 40, 40, 1024 -> 40, 40, 512
    P5_upsample = C3(P5_upsample, int(base_channels * 4), 1, shortcut = False, name = 'conv3_for_upsample1')

    # 40, 40, 512 -> 40, 40, 256
    P4          = DarknetConv2D_BN_SiLU(int(base_channels * 2), (1, 1), name = 'conv_for_feat2')(P5_upsample)
    # 40, 40, 256 -> 80, 80, 256
    P4_upsample = UpSampling2D()(P4)
    # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
    P4_upsample = Concatenate(axis = -1)([P4_upsample, feat1])
    # 80, 80, 512 -> 80, 80, 256
    P4_upsample      = C3(P4_upsample, int(base_channels * 2), 1, shortcut = False, name = 'conv3_for_upsample2')

    P3          = DarknetConv2D_BN_SiLU(int(base_channels * 2), (1, 1), name = 'conv_for_feat1')(P4_upsample)
    # 40, 40, 256 -> 80, 80, 256
    P3_upsample = UpSampling2D()(P3)
    P3_out      = C3(P3_upsample, int(base_channels * 2), 1, shortcut = False, name = 'conv3_for_upsample3')

    # # 80, 80, 256 -> 40, 40, 256
    P3_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    P3_downsample   = DarknetConv2D_BN_SiLU(int(base_channels * 2), (3, 3), strides = (2, 2), name = 'down_sample1')(P3_downsample)
    # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
    P3_downsample   = Concatenate(axis = -1)([P3_downsample, P3])
    # 40, 40, 512 -> 40, 40, 512
    P4_out          = C3(P3_downsample, int(base_channels * 2), 1, shortcut = False, name = 'conv3_for_downsample1') 

    # len(anchors_mask[2]) = 3
    # 5 + num_classes -> 4 + 1 + num_classes
    # 4是先验框的回归系数，1是sigmoid将值固定到0-1，num_classes用于判断先验框是什么类别的物体
    # bs, 20, 20, 3 * (4 + 1 + num_classes)
    out0 = DarknetConv2D(len(anchors_mask[0]) * (8 + nb_classes), (1, 1), strides = (1, 1), name = 'yolo_head_P4')(P4_out)
    out1 = DarknetConv2D(len(anchors_mask[1]) * (8 + nb_classes), (1, 1), strides = (1, 1), name = 'yolo_head_P3')(P3_out)



    pillar_net = tf.keras.models.Model([input_pillars, input_indices], [out0,out1])
    # print(pillar_net.summary())

    return pillar_net

   