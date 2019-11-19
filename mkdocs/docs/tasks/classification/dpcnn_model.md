## class DPCNN_Model
This implementation of DPCNN requires a clear declared sequence length. So sequences input in should be padded or cut to a given length in advance.


### get\_default\_hyper\_parameters
```python
def get_default_hyper_parameters(cls)
```

### downsample
```python
def downsample(inputs, pool_type, sorted, stage)
```

### conv\_block
```python
def conv_block(inputs, filters, kernel_size, activation, shortcut)
```

### resnet\_block
```python
def resnet_block(inputs, filters, kernel_size, activation, shortcut, pool_type, sorted, stage)
```

### build\_model\_arc
```python
def build_model_arc()
```

