"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: bilstm_train.py
@time: 2019-11-20 14:20:17
"""

import band
from band.corpus import SMP2018ECDTCorpus
from band.tasks.classification import BiLSTM_Model
from band.callbacks import EvalCallBack
from band import utils


# Dataset
train_x, train_y = SMP2018ECDTCorpus.load_data('train')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
test_x, test_y = SMP2018ECDTCorpus.load_data('test')

model = BiLSTM_Model()

eval_callback = EvalCallBack(kash_model=model,
                             valid_x=valid_x,
                             valid_y=valid_y,
                             step=5)

model.fit(train_x,
          train_y,
          valid_x,
          valid_y,
          batch_size=32,
          callbacks=[eval_callback])

model.evaluate(test_x, test_y)

# Save model to `saved_classification_model` dir
model.save('saved_classification_model')

# Load model
loaded_model = band.utils.load_model('saved_classification_model')

# Use model to predict
loaded_model.predict(test_x[:10])

# Save model
utils.convert_to_saved_model(model,
                             model_path='saved_model/bilstm',
                             version=1)

