# 基于Pytorch的Hebbian学习算法实现来训练深度卷积神经网络

神经网络模型在各种数据集上使用Hebbian算法和SGD进行训练，以比较结果。
研究了部分层使用Hebbian学习训练、部分层使用SGD训练的混合模型。
还考虑了一种半监督方法，即使用无监督的Hebbian学习预训练DNN的内部层，然后通过SGD微调最后的线性分类器。这种方法在低样本效率场景中比简单的端到端反向传播训练表现更好。

我们还引入了 `neurolab` 包的初步版本，这是一个简单、可扩展的基于Pytorch的深度学习实验框架，提供了处理实验配置、可重复性、检查点/恢复实验状态、超参数搜索和其他实用功能。

这项工作是 https://github.com/GabrieleLagani/HebbianLearningThesis 和 https://github.com/GabrieleLagani/HebbianPCA 的延续，包含最新的更新和新的Hebbian学习算法：Hebbian WTA变体（k-WTA、soft-WTA等）、Hebbian PCA、Hebbian ICA。

**重要：基于FastHebb的新实现可用**，使Hebbian实验速度提高了20多倍！

要启动实验会话，请输入：
```
PYTHONPATH=<项目根目录> python <项目根目录>/runexp.py --config <点路径到配置对象> --mode <模式> --device <设备> --clearhist --restart
```
其中，`<点路径到配置对象>` 是以点表示法表示的路径，指向包含实验配置参数的字典（可以在代码中的任何地方定义）；`<模式>` 是 `train` 或 `test`；`<设备>` 例如 `cpu` 或 `cuda:0`；`--clearhist` 标志可用于如果不需要保留旧的检查点文件；`--restart` 标志可用于从头开始重新启动实验，而不是从检查点恢复。

你也可以使用：
```
PYTHONPATH=<项目根目录> python <项目根目录>/runstack.py --stack <点路径到堆栈对象> --mode <模式> --device <设备> --clearhist --restart
```
来运行完整的实验堆栈（即一个列表，可以在代码中的任何地方定义）。

示例：
```
python runstack.py --stack stacks.vision.all[cifar10] --mode train --device cuda:0 --clearhist
```

作者：Gabriele Lagani - gabriele.lagani@gmail.com
