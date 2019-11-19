## class BaseLabelingModel
Base Sequence Labeling Model


### get\_default\_hyper\_parameters
```python
def get_default_hyper_parameters(cls)
```

### predict\_entities
```python
def predict_entities(x_data, batch_size, join_chunk, debug_info, predict_kwargs)
```
Gets entities from sequence.


##### Args
* **x_data**: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).

* **batch_size**: Integer. If unspecified, it will default to 32.

* **join_chunk**: str or False,

* **debug_info**: Bool, Should print out the logging info.

* **predict_kwargs**: arguments passed to ``predict()`` function of ``tf.keras.Model``

##### Returns
* **list**: list of entity.


### evaluate
```python
def evaluate(x_data, y_data, batch_size, digits, debug_info)
```
Build a text report showing the main classification metrics.


##### Args
* **x_data**: 

* **y_data**: 

* **batch_size**: 

* **digits**: 

* **debug_info**: 


### build\_model\_arc
```python
def build_model_arc()
```

