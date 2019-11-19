"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: train.py
@time: 2019-11-19 19:59:28
"""

import band
from band.corpus import SMP2018ECDTCorpus
from band.tasks.classification import BiLSTM_Model
from band.embeddings import BERTEmbedding
from band.callbacks import EvalCallBack
from tensorflow.python import keras
from band import utils

# If you have a GPU machine
band.config.use_cudnn_cell = True

# Dataset
train_x, train_y = SMP2018ECDTCorpus.load_data('train')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
test_x, test_y = SMP2018ECDTCorpus.load_data('test')

# BERT model path
bert_model_path = 'D:/bert/chinese_L-12_H-768_A-12'
bert_embed = BERTEmbedding(bert_model_path,
                           task=band.CLASSIFICATION,
                           sequence_length=100)

model = BiLSTM_Model(bert_embed)
tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)

eval_callback = EvalCallBack(kash_model=model,
                             valid_x=valid_x,
                             valid_y=valid_y,
                             step=5)

model.fit(train_x,
          train_y,
          valid_x,
          valid_y,
          batch_size=100,
          callbacks=[eval_callback, tf_board_callback])

model.evaluate(test_x, test_y)

# Save model to `saved_classification_model` dir
model.save('saved_classification_model')

# Load model
loaded_model = band.utils.load_model('saved_classification_model')

# Use model to predict
loaded_model.predict(test_x[:10])

# Save model
utils.convert_to_saved_model(model,
                             model_path='saved_model/blstm',
                             version=1)