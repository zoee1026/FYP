import numpy as np
import pandas as pd
from tensorflow.keras.backend import tile
# a=np.array([100,200])
# b=np.array([200,400])
# print(int((b/a)[0]))
y_map=np.loadtxt(r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\yolo\map.csv", delimiter=",")

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
y_map=y_map[::2,::2]
print(y_map.shape)
# use tile to duplicate the array
duplicated_arr = tile(y_map.reshape(*y_map.shape,1, 1), (1, 1, 2, 1))

# check the shape of the duplicated array
print(duplicated_arr.shape)  # output: (256, 320, 3, 4)
