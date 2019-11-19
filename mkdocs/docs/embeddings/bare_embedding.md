## class BareEmbedding
Embedding layer without pre-training, train embedding layer while training model


### \_\_init\_\_
```python
def __init__(task, sequence_length, embedding_size, processor, from_saved_model)
```
Init bare embedding (embedding without pre-training)


##### Args
* **sequence_length**: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
    as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
    various length of input, it will use the length of max sequence in every batch for sequence length.
    If using an integer, let's say ``50``, the input output sequence length will set to 50.

* **embedding_size**: Dimension of the dense embedding.


