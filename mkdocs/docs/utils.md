### unison\_shuffled\_copies
```python
def unison_shuffled_copies(a, b)
```

### get\_list\_subset
```python
def get_list_subset(target, index_list)
```

### custom\_object\_scope
```python
def custom_object_scope()
```

### load\_model
```python
def load_model(model_path, load_weights)
```
Load saved model from saved model from `model.save` function Args: model_path: model folder path load_weights: only load model structure and vocabulary when set to False, default True.



### load\_processor
```python
def load_processor(model_path)
```
Load processor from model When we using tf-serving, we need to use model's processor to pre-process data Args: model_path



### convert\_to\_saved\_model
```python
def convert_to_saved_model(model, model_path, version, inputs, outputs)
```
Export model for tensorflow serving Args: model: Target model model_path: The path to which the SavedModel will be stored. version: The model version code, default timestamp inputs: dict mapping string input names to tensors. These are added to the SignatureDef as the inputs. outputs:  dict mapping string output names to tensors. These are added to the SignatureDef as the outputs.



