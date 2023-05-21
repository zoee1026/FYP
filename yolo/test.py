import numpy as np
import pandas as pd

# a=np.array([100,200])
# b=np.array([200,400])
# print(int((b/a)[0]))
# y_map=np.loadtxt(r"C:\Users\Chan Kin Yan\Documents\GitHub\FYP\yolo\map.csv", delimiter=",")

# ar=y_map[::2, ::2]
# print(ar.shape)

import tensorflow as tf

# Create three Keras tensors with shape (3, 4, 1)
tensor1 = tf.keras.backend.variable(tf.ones((3, 4, 1)))
tensor2 = tf.keras.backend.variable(tf.ones((3, 4, 1)))
tensor3 = tf.keras.backend.variable(tf.ones((3, 4, 1)))

# Concatenate the tensors along the last axis
concatenated_tensor = tf.keras.backend.concatenate([tensor1, tensor2, tensor3], axis=-1)

# Print the shape of the concatenated tensor
print(concatenated_tensor.shape)
