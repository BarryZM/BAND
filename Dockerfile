FROM tensorflow/tensorflow:1.13.1-gpu-py3
# define path 
ARG bert_data='/root/Bert/data'
ARG bert_output='/root/Bert/output'
ARG bert_models='/root/Bert/models'
ARG bert='/root/Bert'
ARG work_space='/root/work'
# mkdir
WORKDIR $work_space
WORKDIR $bert
WORKDIR $bert_data
WORKDIR $bert_output
WORKDIR $bert_models
# download bert pretrain model
RUN curl -O https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip && unzip chinese_L-12_H-768_A-12.zip && rm chinese_L-12_H-768_A-12.zip
# github code
WORKDIR $work_space
ADD ./src ./src
WORKDIR $work_space/src
RUN pip install -r requirements.txt
# env
ENV BERT_BASE_DIR=$bert_models/chinese_L-12_H-768_A-12 
ENV GLUE_DIR=$bert_data 
ENV MODEL_DIR=$bert_output 
ENV MODEL_PB_DIR=$bert/api/