# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 2019-05-23 14:05

import tensorflow as tf
from tensorflow.python import keras
from band.layers.non_masking_layer import NonMaskingLayer
from band.layers.att_wgt_avg_layer import AttentionWeightedAverageLayer
from band.layers.att_wgt_avg_layer import AttentionWeightedAverage, AttWgtAvgLayer
from band.layers.kmax_pool_layer import KMaxPoolingLayer, KMaxPoolLayer, KMaxPooling

L = keras.layers

if __name__ == "__main__":
    print("Hello world")
