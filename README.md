# BiliBili中文标点符号恢复

## 项目结构
project/
│
├── data/
│ └── reader.py
│
├── data_loader/
│ └── loader.py
│
├── model/
│ └── model.py
│
├── predictor/
│ └── predictor.py
│
├── run.py
├── requirements.txt
└── README.md
### 各个组件的描述

- **data/reader.py**: 包含 `PuncRestoreReader` 类，用于读取和处理数据集。
- **data_loader/loader.py**: 包含构建数据加载器的函数。
- **model/model.py**: 定义了 `PuncRestoreLabeler` 模型。
- **predictor/predictor.py**: 包含 `PuncPredictor` 类，用于进行预测。
- **run.py**: 训练模型和进行预测的主脚本。
- **requirements.txt**: 列出了所有需要的依赖库。
- **README.md**: 项目的概述文件。

## 依赖库

CUDA 11.1 

NVIDIA 驱动版本 >= 455.23

运行此项目需要以下依赖库：

- PyTorch >= 1.7.0
- AllenNLP >= 2.5.0
- scikit-learn
- argparse
- pandas
- numpy

你可以使用以下命令来安装依赖库：

```bash
pip install -r requirements.txt
```

## 运行项目

### 训练模型

要训练模型，请使用以下命令：

```
python run.py --mode train
```

要使用训练好的模型进行预测，请使用以下命令：

```
python run.py --mode pred
```

