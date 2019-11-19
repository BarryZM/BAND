# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: __init__.py
@time: 2019-05-17 11:15

"""
import os
os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras_bert
from bang.macros import TaskType, config

custom_objects = keras_bert.get_custom_objects()
CLASSIFICATION = TaskType.CLASSIFICATION
LABELING = TaskType.LABELING

from bang.version import __version__

from bang import layers
from bang import corpus
from bang import embeddings
from bang import macros
from bang import processors
from bang import tasks
from bang import utils
from bang import callbacks

from bang import migeration

migeration.show_migration_guide()
