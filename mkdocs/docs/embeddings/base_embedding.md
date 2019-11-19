## class Embedding
Base class for Embedding Model


### info
```python
def info()
```

### \_\_init\_\_
```python
def __init__(task, sequence_length, embedding_size, processor, from_saved_model)
```

### token\_count
```python
def token_count()
```
corpus token count



### sequence\_length
```python
def sequence_length()
```
model sequence length



### label2idx
```python
def label2idx()
```
label to index dict



### token2idx
```python
def token2idx()
```
token to index dict



### tokenizer
```python
def tokenizer()
```

### sequence\_length
```python
def sequence_length(val)
```

### analyze\_corpus
```python
def analyze_corpus(x, y)
```
Prepare embedding layer and pre-processor for labeling task


##### Args
* **x**: 

* **y**: 


### embed\_one
```python
def embed_one(sentence)
```
Convert one sentence to vector


##### Args
* **sentence**: target sentence, list of str

##### Returns

### embed
```python
def embed(sentence_list, debug)
```
batch embed sentences


##### Args
* **sentence_list**: Sentence list to embed

* **debug**: show debug info

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

### process\_y\_dataset
```python
def process_y_dataset(data, subset)
```
batch process labels data while training


##### Args
* **data**: target dataset

* **subset**: subset index list

##### Returns

### reverse\_numerize\_label\_sequences
```python
def reverse_numerize_label_sequences(sequences, lengths)
```

### \_\_repr\_\_
```python
def __repr__()
```

### \_\_str\_\_
```python
def __str__()
```

