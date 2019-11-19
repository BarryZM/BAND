## class BERTEmbedding
Pre-trained BERT embedding


### info
```python
def info()
```

### \_\_init\_\_
```python
def __init__(model_folder, layer_nums, trainable, task, sequence_length, processor, from_saved_model)
```
Args: task: model_folder: layer_nums: number of layers whose outputs will be concatenated into a single tensor, default `4`, output the last 4 hidden layers as the thesis suggested trainable: whether if the model is trainable, default `False` and set it to `True` for fine-tune this embedding layer during your training sequence_length: processor: from_saved_model



### analyze\_corpus
```python
def analyze_corpus(x, y)
```
Prepare embedding layer and pre-processor for labeling task


##### Args
* **x**: 

* **y**: 


### embed
```python
def embed(sentence_list, debug)
```
batch embed sentences


##### Args
* **sentence_list**: Sentence list to embed

* **debug**: show debug log

##### Returns

### process\_x\_dataset
```python
def process_x_dataset(data, subset)
```
batch process feature data while training


##### Args
* **data**: target dataset

* **subset**: subset index list

##### Returns

