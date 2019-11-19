## class BaseClassificationModel
### \_\_init\_\_
```python
def __init__(embedding, hyper_parameters)
```

### get\_default\_hyper\_parameters
```python
def get_default_hyper_parameters(cls)
```

### build\_model\_arc
```python
def build_model_arc()
```

### compile\_model
```python
def compile_model(**kwargs)
```

### predict
```python
def predict(x_data, batch_size, multi_label_threshold, debug_info, predict_kwargs)
```
Generates output predictions for the input samples.

Computation is done in batches.
##### Args
* **x_data**: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).

* **batch_size**: Integer. If unspecified, it will default to 32.

* **multi_label_threshold**: 

* **debug_info**: Bool, Should print out the logging info.

* **predict_kwargs**: arguments passed to ``predict()`` function of ``tf.keras.Model``

##### Returns

### predict\_top\_k\_class
```python
def predict_top_k_class(x_data, top_k, batch_size, debug_info, predict_kwargs)
```
Generates output predictions with confidence for the input samples.

Computation is done in batches.
##### Args
* **x_data**: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).

* **top_k**: int

* **batch_size**: Integer. If unspecified, it will default to 32.

* **debug_info**: Bool, Should print out the logging info.

* **predict_kwargs**: arguments passed to ``predict()`` function of ``tf.keras.Model``

##### Returns
* **single-label classification**: [
    {
      "label"

* **multi-label classification**: [
    {
      "candidates"


### evaluate
```python
def evaluate(x_data, y_data, batch_size, digits, output_dict, debug_info)
```

