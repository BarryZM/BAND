## class AttentionWeightedAverageLayer
Computes a weighted average of the different channels across timesteps. Uses 1 parameter pr. channel to compute the attention value for a single timestep.


### \_\_init\_\_
```python
def __init__(return_attention, **kwargs)
```

### build
```python
def build(input_shape)
```

### call
```python
def call(x, mask)
```

### get\_output\_shape\_for
```python
def get_output_shape_for(input_shape)
```

### compute\_output\_shape
```python
def compute_output_shape(input_shape)
```

### compute\_mask
```python
def compute_mask(inputs, input_mask)
```

### get\_config
```python
def get_config()
```

