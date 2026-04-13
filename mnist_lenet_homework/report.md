# 使用 CNN（LeNet）对 MNIST 手写数字分类实验报告

## 1. 作业题目
使用 CNN 对 MNIST 手写数字图像进行分类。阅读 LeNet 架构，并用 PyTorch 实现 LeNet，完成训练与测试。

## 2. 模型结构说明
本实验实现了经典 LeNet 风格网络，主要结构如下：
- 卷积层1：`Conv2d(1, 6, kernel_size=5)`
- 池化层1：`AvgPool2d(2, 2)`
- 卷积层2：`Conv2d(6, 16, kernel_size=5)`
- 池化层2：`AvgPool2d(2, 2)`
- 全连接层：`256 -> 120 -> 84 -> 10`

说明：原始 LeNet-5 常见于 `32x32` 输入；MNIST 是 `28x28`。本实现没有强行 resize 到 32x32，而是直接按 28x28 输入计算特征尺寸，所以第二次池化后是 `16x4x4`，展平后输入全连接层为 `256`。

## 3. 关键代码说明
- `model.py`
  - 定义 `LeNet` 网络结构。
  - 在注释中给出每层输出尺寸，便于理解输入尺寸变化。
- `train_lenet.py`
  - 自动选择训练设备（GPU 优先，否则 CPU）。
  - 使用 `torchvision.datasets.MNIST(root='./data', download=True)` 下载数据。
  - 每个 epoch 输出并记录 `Train Loss` 与 `Test Accuracy`。
  - 日志保存到 `logs/train_log.txt`。

## 4. 训练结果记录位置
- 控制台输出：训练过程中每个 epoch 的损失和测试准确率。
- 文件日志：`logs/train_log.txt`。

## 5. 运行结果（请提交时保留）
- 设备：未完成训练（环境缺少 PyTorch）
- 最终测试准确率：暂无（训练未启动）
- 各 epoch 结果摘要：暂无
- 说明：已按要求执行训练命令，但当前容器无法安装 `torch/torchvision`（代理 403），详细错误记录在 `logs/train_log.txt`。

## 6. 结果简要理解
LeNet 结构比较基础，但在 MNIST 这种相对简单的数据集上效果仍然不错。随着 epoch 增加，训练损失通常会下降，测试准确率会上升并趋于稳定。说明卷积网络能够有效提取数字图像中的局部特征，并完成分类任务。
