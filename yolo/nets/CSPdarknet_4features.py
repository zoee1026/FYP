from functools import wraps,reduce

import tensorflow as tf
from keras import backend as K
from keras.initializers import random_normal
from keras.layers import (Add, BatchNormalization, Concatenate, Conv2D, Layer,
                          MaxPooling2D, ZeroPadding2D)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
    
class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class Focus(Layer):
    def __init__(self):
        super(Focus, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] != None else input_shape[1], input_shape[2] // 2 if input_shape[2] != None else input_shape[2], input_shape[3] * 4)

    def call(self, x):
        return tf.concat(
            [x[...,  ::2,  ::2, :],
             x[..., 1::2,  ::2, :],
             x[...,  ::2, 1::2, :],
             x[..., 1::2, 1::2, :]],
             axis=-1
        )

#------------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + SiLU
#---------------------------------------------------#
def DarknetConv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(momentum = 0.97, epsilon = 0.001, name = kwargs['name'] + '.bn'),
        SiLU())

def Bottleneck(x, out_channels, shortcut=True, name = ""):
    y = compose(
            DarknetConv2D_BN_SiLU(out_channels, (1, 1), name = name + '.cv1'),
            DarknetConv2D_BN_SiLU(out_channels, (3, 3), name = name + '.cv2'))(x)
    if shortcut:
        y = Add()([x, y])
    return y

def C3(x, num_filters, num_blocks, shortcut=True, expansion=0.5, name=""):
    hidden_channels = int(num_filters * expansion)
    #----------------------------------------------------------------#
    #   主干部分会对num_blocks进行循环，循环内部是残差结构。
    #----------------------------------------------------------------#
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), name = name + '.cv1')(x)
    #--------------------------------------------------------------------#
    #   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的残差结构
    #--------------------------------------------------------------------#
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), name = name + '.cv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, name = name + '.m.' + str(i))
    #----------------------------------------------------------------#
    #   将大残差边再堆叠回来
    #----------------------------------------------------------------#
    route = Concatenate()([x_1, x_2])

    #----------------------------------------------------------------#
    #   最后对通道数进行整合
    #----------------------------------------------------------------#
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), name = name + '.cv3')(route)

def SPPBottleneck(x, out_channels, name = ""):
    #---------------------------------------------------#
    #   使用了SPP结构，即不同尺度的最大池化后堆叠。
    #---------------------------------------------------#
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), name = name + '.cv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), name = name + '.cv2')(x)
    return x
    
def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, name = ""):
    #----------------------------------------------------------------#
    #   利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
    #----------------------------------------------------------------#

    # 320, 320, 64 => 160, 160, 128
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), name = name + '.0')(x)
    if last:
        x = SPPBottleneck(x, num_filters, name = name + '.1')
    return C3(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, name = name + '.1' if not last else name + '.2')

#---------------------------------------------------#
#   CSPdarknet的主体部分
#   输入为一张640x640x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
def darknet_body(x, base_channels, base_depth):
    #---------------------------------------------------#
    #   base_channels 默认值为64
    #---------------------------------------------------#
    # 640, 640, 3 => 320, 320, 12
    # x = Focus()(x)
    
    # 320, 320, 12 => 320, 320, 64
    x = DarknetConv2D_BN_SiLU(base_channels, (3, 3), name = 'backbone.stem.conv')(x)
    
    # 320, 320, 64 => 160, 160, 128
    x = resblock_body(x, base_channels * 2, base_depth, name = 'backbone.dark2')
    feat1 = x
    
    # 160, 160, 128 => 80, 80, 256
    x = resblock_body(x, base_channels * 4, base_depth * 3, name = 'backbone.dark3')
    feat2 = x
    
    # 80, 80, 256 => 40, 40, 512
    x = resblock_body(x, base_channels * 8, base_depth * 3, name = 'backbone.dark4')
    feat3 = x
    
    # 40, 40, 512 => 20, 20, 1024
    x = resblock_body(x, base_channels * 16, base_depth, shortcut=False, last=True, name = 'backbone.dark5')
    feat4 = x
    return feat1, feat2, feat3, feat4

