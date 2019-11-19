## class CRF
Conditional Random Field layer (tf.keras) `CRF` can be used as the last layer in a network (as a classifier). Input shape (features) must be equal to the number of classes the CRF can predict (a linear layer is recommended). Note: the loss and accuracy functions of networks using `CRF` must use the provided loss and accuracy functions (denoted as loss and viterbi_accuracy) as the classification of sequences are used with the layers internal weights. Args: output_dim (int): the number of labels to tag each temporal input. Input shape: nD tensor with shape `(batch_size, sentence length, num_classes)`. Output shape: nD tensor with shape: `(batch_size, sentence length, num_classes)`.


### \_\_init\_\_
```python
def __init__(output_dim, mode, supports_masking, transitions, **kwargs)
```

### get\_config
```python
def get_config()
```

### build
```python
def build(input_shape)
```

### call
```python
def call(inputs, **kwargs)
```

### loss
```python
def loss(y_true, y_pred)
```

### compute\_output\_shape
```python
def compute_output_shape(input_shape)
```

### viterbi\_accuracy
```python
def viterbi_accuracy()
```

