# BAND：BERT Application aNd Deployment

A simple and efficient BERT model training and deployment framework，一个简单高效的 BERT 模型训练和部署框架

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/SunYanCN/BAND">
    <img src="figures/logo.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">BAND</h3>
  <p align="center">
    BAND：BERT Application aNd Deployment
    <br />
    <a href="https://sunyancn.github.io/BAND/"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/SunYanCN/BAND/tree/master/examples">查看Demo</a>
    ·
    <a href="https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/issues/new?assignees=&labels=&template=bug_report.md&title=">报告Bug</a>
    ·
    <a href="https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/issues/new?assignees=&labels=&template=feature_request.md&title=">提出新特性</a>
        ·
    <a href="https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/issues/new?assignees=&labels=&template=custom.md&title=">问题交流</a>
  </p>

</p>
 
## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装方法](#安装方法)
- [文件目录说明](#文件目录说明)
- [开发的架构](#开发的架构)
- [部署](#部署)
- [使用到的框架](#使用到的框架)
- [贡献者](#贡献者)
  - [如何参与开源项目](#如何参与开源项目)
- [版本控制](#版本控制)
- [作者](#作者)
- [鸣谢](#鸣谢)

### 上手指南

###### **开发前的配置要求**

1. Linux (Centos,Ubuntu.....)
2. Python>=3.6
3. Tensorflow>=1.13.1

###### **安装方法**
安装band有两种方式：
- Install from PyPi
    ```sh
    pip install band
    ```
- Install From Git
    ```sh
    pip install git+https://www.github.com/sunyancn/band.git
    ```
###### 文本分类Demo
1. 训练模型
    ```python
    import band
    from band.corpus import SMP2018ECDTCorpus
    from band.tasks.classification import BiLSTM_Model
    from band.callbacks import EvalCallBack
    from band import utils
    
    # Dataset
    dataset = SMP2018ECDTCorpus()
    
    model = BiLSTM_Model()
    eval_callback = EvalCallBack(kash_model=model,
                                 valid_x=dataset.valid_x,
                                 valid_y=dataset.valid_y,
                                 step=5)
    model.fit(dataset.train_x,
              dataset.train_y,
              dataset.valid_x,
              dataset.valid_y,
              batch_size=32,
              callbacks=[eval_callback])
    
    model.evaluate(dataset.test_x, dataset.test_y)
    
    # Save model to `saved_classification_model` dir
    model.save('saved_classification_model')
    
    # Load model
    loaded_model = band.utils.load_model('saved_classification_model')
    
    # Use model to predict
    loaded_model.predict(dataset.test_x[:10])
    
    # Save model
    utils.convert_to_saved_model(model,
                                 model_path='saved_model/bilstm',
                                 version='1')
    ```

2. 部署模型
    ```bash
    simple_tensorflow_serving --model_base_path="saved_model/bilstm"
    ```

3. 启动WebAPP,参考[代码](https://github.com/SunYanCN/BAND/tree/master/webapp)
    ```
    python app.py
    ```
4. 演示
    <div align=center><img src="https://s2.ax1x.com/2019/11/21/MonlUU.gif" width="800"/></div>
   
### 开发的架构

<div align=center><img src="https://s2.ax1x.com/2019/11/20/Mf2YAU.md.png" width="500"/></div>

### 部署

暂无

### 使用到的框架

- [TensorFlow](https://getbootstrap.com)
- [simple-tensorflow-serving](https://stfs.readthedocs.io/en/latest/index.html)

### 作者
您可以通过以下方式联系我：
- **Email**: sunyanhust@163.com
- **NLP技术QQ交流群**：859886087

> 您也可以在贡献者名单中参看所有参与该项目的开发者。


### 贡献者

请阅读**CONTRIBUTING.md** 查阅为该项目做出贡献的开发者。

#### 如何参与开源项目

贡献使开源社区成为一个学习、激励和创造的绝佳场所。你所作的任何贡献都是**非常感谢**的。


1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 版权说明

该项目签署了Apache授权许可，详情请参阅 [LICENSE](https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/blob/master/LICENSE)

### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 鸣谢
- [Kashgari](https://github.com/BrikerMan/Kashgari)
- [bert4keras](https://github.com/bojone/bert4keras)
- [Free Logo Design](https://www.freelogodesign.org/)
- [Headliner](https://github.com/as-ideas/headliner)

<!-- links -->
[your-project-path]: SunYanCN/BERT-chinese-text-classification-and-deployment
[contributors-shield]: https://img.shields.io/github/contributors/SunYanCN/BERT-chinese-text-classification-and-deployment.svg?style=flat-square
[contributors-url]: https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/SunYanCN/BERT-chinese-text-classification-and-deployment.svg?style=flat-square
[forks-url]: https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/network/members
[stars-shield]: https://img.shields.io/github/stars/SunYanCN/BERT-chinese-text-classification-and-deployment.svg?style=flat-square
[stars-url]: https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/stargazers
[issues-shield]: https://img.shields.io/github/issues/SunYanCN/BERT-chinese-text-classification-and-deployment.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/SunYanCN/BERT-chinese-text-classification-and-deployment.svg
[license-shield]: https://img.shields.io/github/license/SunYanCN/BERT-chinese-text-classification-and-deployment.svg?style=flat-square
[license-url]: https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/blob/master/LICENSE
