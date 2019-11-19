## class NonMaskingLayer
fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978 thanks for https://github.com/jacoxu


### \_\_init\_\_
```python
def __init__(**kwargs)
```

### build
```python
def build(input_shape)
```

### compute\_mask
```python
def compute_mask(inputs, input_mask)
```

### call
```python
def call(x, mask)
```

