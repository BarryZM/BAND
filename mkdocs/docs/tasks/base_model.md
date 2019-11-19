## class BaseModel
Base Sequence Labeling Model


### get\_default\_hyper\_parameters
```python
def get_default_hyper_parameters(cls)
```

### info
```python
def info()
```

### task
```python
def task()
```

### token2idx
```python
def token2idx()
```

### label2idx
```python
def label2idx()
```

### pre\_processor
```python
def pre_processor()
```

### processor
```python
def processor()
```

### \_\_init\_\_
```python
def __init__(embedding, hyper_parameters)
```
Args: embedding: model embedding hyper_parameters: a dict of hyper_parameters.

Examples: You could change customize hyper_parameters like this::

# get default hyper_parameters hyper_parameters = BLSTMModel.get_default_hyper_parameters() # change lstm hidden unit to 12 hyper_parameters['layer_blstm']['units'] = 12 # init new model with customized hyper_parameters labeling_model = BLSTMModel(hyper_parameters=hyper_parameters) labeling_model.fit(x, y)

### build\_model
```python
def build_model(x_train, y_train, x_validate, y_validate)
```
Build model with corpus


##### Args
* **x_train**: Array of train feature data (if the model has a single input),
    or tuple of train feature data array (if the model has multiple inputs)

* **y_train**: Array of train label data

* **x_validate**: Array of validation feature data (if the model has a single input),
    or tuple of validation feature data array (if the model has multiple inputs)

* **y_validate**: Array of validation label data


### build\_multi\_gpu\_model
```python
def build_multi_gpu_model(gpus, x_train, y_train, cpu_merge, cpu_relocation, x_validate, y_validate)
```
Build multi-GPU model with corpus


##### Args
* **gpus**: Integer >= 2, number of on GPUs on which to create model replicas.

* **cpu_merge**: A boolean value to identify whether to force merging model weights
    under the scope of the CPU or not.

* **cpu_relocation**: A boolean value to identify whether to create the model's weights
    under the scope of the CPU. If the model is not defined under any preceding device
    scope, you can still rescue it by activating this option.

* **x_train**: Array of train feature data (if the model has a single input),
    or tuple of train feature data array (if the model has multiple inputs)

* **y_train**: Array of train label data

* **x_validate**: Array of validation feature data (if the model has a single input),
    or tuple of validation feature data array (if the model has multiple inputs)

* **y_validate**: Array of validation label data


### build\_tpu\_model
```python
def build_tpu_model(strategy, x_train, y_train, x_validate, y_validate)
```
Build TPU model with corpus


##### Args
* **strategy**: `TPUDistributionStrategy`. The strategy to use for replicating model
    across multiple TPU cores.

* **x_train**: Array of train feature data (if the model has a single input),
    or tuple of train feature data array (if the model has multiple inputs)

* **y_train**: Array of train label data

* **x_validate**: Array of validation feature data (if the model has a single input),
    or tuple of validation feature data array (if the model has multiple inputs)

* **y_validate**: Array of validation label data


### get\_data\_generator
```python
def get_data_generator(x_data, y_data, batch_size, shuffle)
```
data generator for fit_generator


##### Args
* **x_data**: Array of feature data (if the model has a single input),
    or tuple of feature data array (if the model has multiple inputs)

* **y_data**: Array of label data

* **batch_size**: Number of samples per gradient update, default to 64.

* **shuffle**: 

##### Returns

### fit
```python
def fit(x_train, y_train, x_validate, y_validate, batch_size, epochs, callbacks, fit_kwargs, shuffle)
```
Trains the model for a given number of epochs with fit_generator (iterations on a dataset).


##### Args
* **x_train**: Array of train feature data (if the model has a single input),
    or tuple of train feature data array (if the model has multiple inputs)

* **y_train**: Array of train label data

* **x_validate**: Array of validation feature data (if the model has a single input),
    or tuple of validation feature data array (if the model has multiple inputs)

* **y_validate**: Array of validation label data

* **batch_size**: Number of samples per gradient update, default to 64.

* **epochs**: Integer. Number of epochs to train the model. default 5.

* **callbacks**: 

* **fit_kwargs**: fit_kwargs

* **shuffle**: 


### fit\_without\_generator
```python
def fit_without_generator(x_train, y_train, x_validate, y_validate, batch_size, epochs, callbacks, fit_kwargs)
```
Trains the model for a given number of epochs (iterations on a dataset).


##### Args
* **x_train**: Array of train feature data (if the model has a single input),
    or tuple of train feature data array (if the model has multiple inputs)

* **y_train**: Array of train label data

* **x_validate**: Array of validation feature data (if the model has a single input),
    or tuple of validation feature data array (if the model has multiple inputs)

* **y_validate**: Array of validation label data

* **batch_size**: Number of samples per gradient update, default to 64.

* **epochs**: Integer. Number of epochs to train the model. default 5.

* **callbacks**: 

* **fit_kwargs**: fit_kwargs


### compile\_model
```python
def compile_model(**kwargs)
```
Configures the model for training.

Using ``compile()`` function of ``tf.keras.Model``

- https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#compile
##### Args
* ****kwargs**: arguments passed to ``compile()`` function of ``tf.keras.Model``

* **ults**: 

* **- loss**: ``categorical_crossentropy``

* **- optimizer**: ``adam``

* **- metrics**: ``['accuracy']``


### predict
```python
def predict(x_data, batch_size, debug_info, predict_kwargs)
```
Generates output predictions for the input samples.

Computation is done in batches.
##### Args
* **x_data**: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).

* **batch_size**: Integer. If unspecified, it will default to 32.

* **debug_info**: Bool, Should print out the logging info.

* **predict_kwargs**: arguments passed to ``predict()`` function of ``tf.keras.Model``

##### Returns

### evaluate
```python
def evaluate(x_data, y_data, batch_size, digits, debug_info)
```
Evaluate model Args: x_data: y_data: batch_size: digits: debug_info



### build\_model\_arc
```python
def build_model_arc()
```

### save
```python
def save(model_path)
```
Save model Args: model_path



