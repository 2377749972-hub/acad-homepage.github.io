# MNIST LeNet 作业项目

这是一个用于课程附加作业的简洁 PyTorch 项目，实现 LeNet 对 MNIST 手写数字分类。

## 1. 项目结构

```text
mnist_lenet_homework/
├── model.py
├── train_lenet.py
├── requirements.txt
├── report.md
└── logs/
```

## 2. 安装依赖

建议使用 Python 3.9+。

```bash
cd mnist_lenet_homework
pip install -r requirements.txt
```

## 3. 运行训练

```bash
python train_lenet.py
```

运行后将会：
- 自动下载 MNIST 数据到 `./data`
- 自动选择 CPU/GPU
- 每个 epoch 输出训练损失和测试准确率
- 保存日志到 `logs/train_log.txt`

## 4. 查看结果

- 训练日志：`logs/train_log.txt`
- 实验说明：`report.md`
