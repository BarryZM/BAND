## class ClassificationProcessor
Corpus Pre Processor class


### \_\_init\_\_
```python
def __init__(multi_label, **kwargs)
```

### info
```python
def info()
```

### process\_y\_dataset
```python
def process_y_dataset(data, max_len, subset)
```

### numerize\_token\_sequences
```python
def numerize_token_sequences(sequences)
```

### numerize\_label\_sequences
```python
def numerize_label_sequences(sequences)
```
Convert label sequence to label-index sequence ``['O', 'O', 'B-ORG'] -> [0, 0, 2]``


##### Args
* **sequences**: label sequence, list of str

##### Returns

### reverse\_numerize\_label\_sequences
```python
def reverse_numerize_label_sequences(sequences, **kwargs)
```

