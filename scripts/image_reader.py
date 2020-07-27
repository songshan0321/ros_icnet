import os
import numpy as np
import tensorflow as tf
import glob
import cv2

def _extract_mean(img, img_mean, swap_channel=False):
    # swap channel and extract mean
    
    if swap_channel:
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)

    img -= img_mean
    
    return img

def _check_input(img):
    ori_h, ori_w = img.get_shape().as_list()[1:3]
    
    if ori_h % 32 != 0 or ori_w % 32 != 0:
        new_h = (int(ori_h/32) + 1) * 32
        new_w = (int(ori_w/32) + 1) * 32
        shape = [new_h, new_w]
        
        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)
        
        print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]

    return img, shape

def _infer_preprocess(img, shape, swap_channel=False):
    o_shape = img.shape[0:2]
        
    img = _extract_mean(img, swap_channel)
    img = tf.expand_dims(img, axis=0)
    img, n_shape = _check_input(img)
        
    return img, o_shape, n_shape