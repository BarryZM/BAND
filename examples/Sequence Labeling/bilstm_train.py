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
from band.corpus import ChineseDailyNerCorpus
from band.tasks.labeling import BiLSTM_Model
from band.embeddings import BareEmbedding
from band.callbacks import EvalCallBack
from band import utils

# Dataset
dataset = ChineseDailyNerCorpus()

bare_embedding = BareEmbedding(task=band.LABELING,sequence_length=50)
model = BiLSTM_Model(bare_embedding)
eval_callback = EvalCallBack(kash_model=model,
                             valid_x=dataset.valid_x,
                             valid_y=dataset.valid_y,
                             step=5)
model.fit(dataset.train_x,
          dataset.train_y,
          dataset.valid_x,
          dataset.valid_y,
          batch_size=32,
          callbacks=[eval_callback])

model.evaluate(dataset.test_x, dataset.test_y)

# Save model to `saved_ner_model` dir
model.save('saved_ner_model')

# Load model
loaded_model = band.utils.load_model('saved_ner_model')

# Use model to predict
loaded_model.predict(dataset.test_x[:10])

# Save model
utils.convert_to_saved_model(model,
                             model_path='saved_model/bilstm',
                             version='1')

