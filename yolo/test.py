import numpy as np
import pandas as pd
from tensorflow.keras.backend import tile
from keras import backend as K

# a=np.array([100,200])
# b=np.array([200,400])
# print(int((b/a)[0]))
# y_map=np.loadtxt(r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\yolo\map.csv", delimiter=",")

# ar=y_map[::2, ::2]
# print(ar.shape)

import tensorflow as tf

# Create three Keras tensors with shape (3, 4, 1)
# tensor1 = tf.keras.backend.variable(tf.ones((3, 4, 1)))
# tensor2 = tf.keras.backend.variable(tf.ones((3, 4, 1)))
# tensor3 = tf.keras.backend.variable(tf.ones((3, 4, 1)))

# # Concatenate the tensors along the last axis
# concatenated_tensor = tf.keras.backend.concatenate([tensor1, tensor2, tensor3], axis=-1)

# # Print the shape of the concatenated tensor
# print(concatenated_tensor.shape)


# create a 3x4 numpy array
# y_map=y_map[::2,::2]
# print(y_map.shape)
# # use tile to duplicate the array
# duplicated_arr = tile(y_map.reshape(*y_map.shape,1, 1), (1, 1, 2, 1))

# # check the shape of the duplicated array
# print(duplicated_arr.shape)  # output: (256, 320, 3, 4)


# create a numpy array with shape (3, 4, 3, 3)
# arr = np.random.rand(3, 4, 3, 3)

# # calculate the sum of squares of the last two values in the array
# last_two_values = arr[:, :, -2:, :]
# sum_of_squares = np.sum(np.square(last_two_values))

# # print the sum of squares
# print("Sum of squares of last two values:", sum_of_squares)

# anchor_path=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\folder_root\AnchorKmeans6.csv'
# anchors=np.round(np.array(pd.read_csv(
#         anchor_path, index_col=0).values, dtype=np.float32).tolist(), 3)[:3]
# anchors_tensor = K.tile(
#         anchors.reshape(1, 1, *anchors.shape), [3, 3, 1, 1])
# print(anchors_tensor)

# def cal_dig(row):
#     return np.sqrt(np.sum([np.power(row[0],2),np.power(row[1],2)]))

# anchors_diag = K.cast(K.map_fn(cal_dig,anchors_tensor),K.dtype(anchors_tensor))
# # anchors_diag=anchors_diag.reshape(*anchors_diag.shape,1)
# # anchors_diag = K.cast(K.sqrt(K.sum(K.square(anchors_tensor[..., 0:2]), axis=1)), K.dtype(anchors_tensor))
# print(anchors_diag.shape)
# print(K.eval(anchors_diag))



# create a Keras tensor with shape (None, 3, 3)
tensor = K.variable(np.random.rand(2, 2,3, 3))

# define a function that takes a tensor and returns its mean along the last dimension
def last_dim_mean(tensor):
    return K.mean(tensor, axis=-1, keepdims=True)

# apply the function to the last dimension of the tensor using K.map_fn
result_tensor = K.map_fn(last_dim_mean, tensor)

# print the resulting tensor
print(K.eval(result_tensor))