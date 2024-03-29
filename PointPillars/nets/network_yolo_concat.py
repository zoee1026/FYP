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

from nets.CSPdarknet_3resnet import (C3, DarknetConv2D, DarknetConv2D_BN_SiLU,
                             darknet_body,compose)
import tensorflow as tf
import numpy as np
from config import Parameters



def build_point_pillar_graph(params: Parameters, batch_size: int = Parameters.batch_size):

    # extract required parameters
    max_pillars = int(params.max_pillars)
    max_points  = int(params.max_points_per_pillar)
    nb_features = int(params.nb_features)
    nb_channels = int(params.nb_channels)
    batch_size  = int(params.batch_size)
    image_size  = tuple([params.Xn, params.Yn])
    nb_classes  = int(params.nb_classes)
    nb_anchors  = len(params.anchor_dims)
    base_depth  = 3

    if tf.keras.backend.image_data_format() == "channels_first":
        raise NotImplementedError
    else:
        input_shape = (max_pillars, max_points, nb_features)

    input_pillars = tf.keras.layers.Input(input_shape, batch_size=batch_size, name="pillars/input")
    input_indices = tf.keras.layers.Input((max_pillars, 3), batch_size=batch_size, name="pillars/indices",
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


    # pillars
    x = tf.keras.layers.Conv2D(nb_channels, (1, 1), activation='linear', use_bias=False, name="pillars/conv2d")(input_pillars)
    x = tf.keras.layers.BatchNormalization(name="pillars/batchnorm", fused=True, epsilon=1e-3, momentum=0.99)(x)
    x = tf.keras.layers.Activation("relu", name="pillars/relu")(x)
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

    feat1, feat2, feat3 = darknet_body(pillars, nb_channels, base_depth)
    # 20, 20, 1024 -> 20, 20, 512
    P5          = DarknetConv2D_BN_SiLU(int(nb_channels * 8), (1, 1), name = 'conv_for_feat3')(feat3)  
    # 20, 20, 512 -> 40, 40, 512
    P5_upsample = UpSampling2D()(P5) 
    # 40, 40, 512 cat 40, 40, 512 -> 40, 40, 1024
    P5_upsample = Concatenate(axis = -1)([P5_upsample, feat2])
    # 40, 40, 1024 -> 40, 40, 512
    P5_upsample = C3(P5_upsample, int(nb_channels * 8), base_depth, shortcut = False, name = 'conv3_for_upsample1')

    # 40, 40, 512 -> 40, 40, 256
    P4          = DarknetConv2D_BN_SiLU(int(nb_channels * 4), (1, 1), name = 'conv_for_feat2')(P5_upsample)
    # 40, 40, 256 -> 80, 80, 256
    P4_upsample = UpSampling2D()(P4)
    # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
    P4_upsample = Concatenate(axis = -1)([P4_upsample, feat1])
    # 80, 80, 512 -> 80, 80, 256
    P3_out      = C3(P4_upsample, int(nb_channels * 4), base_depth, shortcut = False, name = 'conv3_for_upsample2')

    # 80, 80, 256 -> 40, 40, 256
    P3_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    P3_downsample   = DarknetConv2D_BN_SiLU(int(nb_channels * 4), (3, 3), strides = (2, 2), name = 'down_sample1')(P3_downsample)
    # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
    P3_downsample   = Concatenate(axis = -1)([P3_downsample, P4])
    # 40, 40, 512 -> 40, 40, 512
    P4_out          = C3(P3_downsample, int(nb_channels * 8), base_depth, shortcut = False, name = 'conv3_for_downsample1') 

    # 40, 40, 512 -> 20, 20, 512
    P4_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P4_out)
    P4_downsample   = DarknetConv2D_BN_SiLU(int(nb_channels * 8), (3, 3), strides = (2, 2), name = 'down_sample2')(P4_downsample)
    # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
    P4_downsample   = Concatenate(axis = -1)([P4_downsample, P5])
    # 20, 20, 1024 -> 20, 20, 1024
    P5_out          = C3(P4_downsample, int(nb_channels * 16), base_depth, shortcut = False, name = 'conv3_for_downsample2')

    # UpSample back 
    out0 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(4, 4), padding="same", activation="linear",
                                          use_bias=False, name="cnn/up0/conv2dt")(P3_out)
    out0 = tf.keras.layers.BatchNormalization(name="cnn/up3/bn", fused=True)(out0)
    out0 = tf.keras.layers.Activation("relu", name="cnn/up3/relu")(out0)

    out1 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(8, 8), padding="same", activation="linear",
                                          use_bias=False, name="cnn/up1/conv2dt")(P4_out)
    out1 = tf.keras.layers.BatchNormalization(name="cnn/up3/bn", fused=True)(out1)
    out1 = tf.keras.layers.Activation("relu", name="cnn/up3/relu")(out1)

    out2 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(16, 16), padding="same", activation="linear",
                                          use_bias=False, name="cnn/up2conv2dt")(P5_out)
    out2 = tf.keras.layers.BatchNormalization(name="cnn/up3/bn", fused=True)(out2)
    out2 = tf.keras.layers.Activation("relu", name="cnn/up3/relu")(out2)

    concat = tf.keras.layers.Concatenate(name="cnn/concatenate")([out1, out2, out0])

    # len(anchors_mask[2]) = 3
    # 5 + num_classes -> 4 + 1 + num_classes
    # 4是先验框的回归系数，1是sigmoid将值固定到0-1，num_classes用于判断先验框是什么类别的物体
    # bs, 20, 20, 3 * (4 + 1 + num_classes)
    # out2 = DarknetConv2D(len(anchors_mask[2]) * (5 + num_classes), (1, 1), strides = (1, 1), name = 'yolo_head_P3')(P3_out)
    # out1 = DarknetConv2D(len(anchors_mask[1]) * (5 + num_classes), (1, 1), strides = (1, 1), name = 'yolo_head_P4')(P4_out)
    # out0 = DarknetConv2D(len(anchors_mask[0]) * (5 + num_classes), (1, 1), strides = (1, 1), name = 'yolo_head_P5')(P5_out)
   
    # Detection head
    occ = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="occupancy", activation="sigmoid")(concat)

    loc = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="loc", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
    loc = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="loc/reshape")(loc)

    size = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="size", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
    size = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="size/reshape")(size)

    angle = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="angle")(concat)

    heading = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="heading", activation="sigmoid")(concat)

    clf = tf.keras.layers.Conv2D(nb_anchors * nb_classes, (1, 1), name="clf", activation="linear")(concat)
    clf = tf.keras.layers.Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="clf/reshape")(clf)

    pillar_net = tf.keras.models.Model([input_pillars, input_indices], [occ, loc, size, angle, heading, clf])
    print(pillar_net.summary())

    return pillar_net

   