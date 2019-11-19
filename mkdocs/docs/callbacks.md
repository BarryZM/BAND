## class EvalCallBack
### \_\_init\_\_
```python
def __init__(kash_model, valid_x, valid_y, step, batch_size, average)
```
Evaluate callback, calculate precision, recall and f1 Args: kash_model: the band model to evaluate valid_x: feature data valid_y: label data step: step, default 5 batch_size: batch size, default 256



### on\_epoch\_end
```python
def on_epoch_end(epoch, logs)
```

