## class WordEmbedding
Pre-trained word2vec embedding


### info
```python
def info()
```

### \_\_init\_\_
```python
def __init__(w2v_path, task, w2v_kwargs, sequence_length, processor, from_saved_model)
```
Args: task: w2v_path: word2vec file path w2v_kwargs: params pass to the ``load_word2vec_format()`` function of ``gensim.models.KeyedVectors`` - https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length as sequence length. When using ``'variable'``, model input shape will set to None, which can handle various length of input, it will use the length of max sequence in every batch for sequence length. If using an integer, let's say ``50``, the input output sequence length will set to 50. processor



### analyze\_corpus
```python
def analyze_corpus(x, y)
```
Prepare embedding layer and pre-processor for labeling task


##### Args
* **x**: 

* **y**: 


