## class KMaxPoolingLayer
K-max pooling layer that extracts the k-highest activation from a sequence (2nd dimension). TensorFlow backend. # Arguments k: An int scale, indicate k max steps of features to pool. sorted: A bool, if output is sorted (default) or not. data_format: A string, one of `channels_last` (default) or `channels_first`. The ordering of the dimensions in the inputs. `channels_last` corresponds to inputs with shape `(batch, steps, features)` while `channels_first` corresponds to inputs with shape `(batch, features, steps)`. # Input shape - If `data_format='channels_last'`: 3D tensor with shape: `(batch_size, steps, features)` - If `data_format='channels_first'`: 3D tensor with shape: `(batch_size, features, steps)` # Output shape 3D tensor with shape: `(batch_size, top-k-steps, features)`


### \_\_init\_\_
```python
def __init__(k, sorted, data_format, **kwargs)
```

### compute\_output\_shape
```python
def compute_output_shape(input_shape)
```

### call
```python
def call(inputs)
```

### get\_config
```python
def get_config()
```

