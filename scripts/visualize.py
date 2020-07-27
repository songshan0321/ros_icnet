import scipy.io as sio
import numpy as np
import tensorflow as tf

# --------------- Set label color here -------------------
label_colours = [[255, 255, 255], [0, 0, 0]]
                # 0 = free, 1 = not free
# --------------------------------------------------------


def decode_labels(mask, img_shape, num_classes):
    color_table = label_colours

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3))
    
    return pred
