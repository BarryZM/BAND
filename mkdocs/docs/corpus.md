## class DataReader
### read\_conll\_format\_file
```python
def read_conll_format_file(file_path, text_index, label_index)
```
Read conll format data_file Args: file_path: path of target file text_index: index of text data, default 0 label_index: index of label data, default 1



## class ChineseDailyNerCorpus
Chinese Daily New New Corpus https://github.com/zjy-ucas/ChineseNER/


### load\_data
```python
def load_data(cls, subset_name, shuffle)
```
Load dataset as sequence labeling format, char level tokenized

Sample::

train_x, train_y = ChineseDailyNerCorpus.load_data('train') test_x, test_y = ChineseDailyNerCorpus.load_data('test')
##### Args
* **subset_name**: {train, test, valid}

* **shuffle**: should shuffle or not, default True.

##### Returns

## class CONLL2003ENCorpus
### load\_data
```python
def load_data(cls, subset_name, task_name, shuffle)
```




## class SMP2018ECDTCorpus
https://worksheets.codalab.org/worksheets/0x27203f932f8341b79841d50ce0fd684f/

This dataset is released by the Evaluation of Chinese Human-Computer Dialogue Technology (SMP2018-ECDT) task 1 and is provided by the iFLYTEK Corporation, which is a Chinese human-computer dialogue dataset.
### load\_data
```python
def load_data(cls, subset_name, shuffle, cutter)
```
Load dataset as sequence classification format, char level tokenized Samples:: train_x, train_y = SMP2018ECDTCorpus.load_data('train') test_x, test_y = SMP2018ECDTCorpus.load_data('test')


##### Args
* **subset_name**: {train, test, valid}

* **shuffle**: should shuffle or not, default True.

* **cutter**: sentence cutter, {char, jieba}

##### Returns

