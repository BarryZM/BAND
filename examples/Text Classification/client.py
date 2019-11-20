import requests
from band import utils
import numpy as np


def main():
    endpoint = "http://127.0.0.1:8500"
    x = ['这', '个', '价', '不', '算', '高', '，', '和', '一', '天', '内', '训', '相', '比', '相', '差', '无', '几']
    processor = utils.load_processor(model_path='saved_model/bilstm/1')
    tensor = processor.process_x_dataset([x])
    json_data = {"model_name": "default", "data": {"input:0": tensor.tolist()}}
    result = requests.post(endpoint, json=json_data)
    preds = dict(result.json())['dense/Softmax:0']
    label_index = np.array(preds).argmax(-1)
    labels = processor.reverse_numerize_label_sequences(label_index)
    print(labels)


if __name__ == "__main__":
    main()
