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
    nb_anchors  = len(params.anchor_dims)
    base_depth  = 3

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


    # pillars
    # x = tf.keras.layers.Conv2D(nb_channels, (1, 1), activation='linear', use_bias=False, name="pillars/conv2d")(input_pillars)
    # x = tf.keras.layers.BatchNormalization(name="pillars/batchnorm", fused=True, epsilon=1e-3, momentum=0.99)(x)
    # x = tf.keras.layers.Activation("relu", name="pillars/relu")(x)
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
    P3_out      = C3(P4_upsample, int(base_channels * 2), 1, shortcut = False, name = 'conv3_for_upsample2')


    # upsampling
    output = tf.keras.layers.Conv2DTranspose(int(base_channels * 2), (3, 3), strides=(2, 2), padding="same", activation="linear",
                                          use_bias=False, name="convTransup")(P3_out)
    output = tf.keras.layers.BatchNormalization(name="convTransup/bn", fused=True)(output)
    output = tf.keras.layers.Activation("relu", name="convTransup/relu")(output)
  
    # Detection head
    occ = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="occupancy", activation="sigmoid")(output)

    loc = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="loc", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(output)
    loc = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="loc/reshape")(loc)

    size = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="size", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(output)
    size = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="size/reshape")(size)

    angle = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="angle")(output)

    heading = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="heading", activation="sigmoid")(output)

    clf = tf.keras.layers.Conv2D(nb_anchors * nb_classes, (1, 1), name="clf", activation="linear")(output)
    clf = tf.keras.layers.Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="clf/reshape")(clf)

    pillar_net = tf.keras.models.Model([input_pillars, input_indices], [occ, loc, size, angle, heading, clf])
    print(pillar_net.summary())

    return pillar_net

   