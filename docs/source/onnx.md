# torch.onnx

## 概述

[开放神经网络交换格式 (ONNX)](https://onnx.ai/) 是一种用于表示机器学习模型的开放标准格式。`torch.onnx` 模块能够从原生的 PyTorch `torch.nn.Module` 模型中捕获计算图，并将其转换为 [ONNX 图](https://github.com/onnx/onnx/blob/main/docs/IR.md)。

导出的模型可以被众多 [支持 ONNX 的运行时](https://onnx.ai/supported-tools.html#deployModel) 中的任何一个使用，包括微软的 [ONNX Runtime](https://www.onnxruntime.ai)。

下面的示例展示了如何导出一个简单的模型。

```python
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))

input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

model = MyModel()

torch.onnx.export(
    model,                  # 要导出的模型
    (input_tensor,),        # 模型的输入
    "my_model.onnx",        # ONNX 模型的文件名
    input_names=["input"],  # 为 ONNX 模型重命名输入
    dynamo=True             # True 或 False 来选择要使用的导出器
)
```

## 基于 torch.export 的 ONNX 导出器

*基于 torch.export 的 ONNX 导出器是 PyTorch 2.6 及更新版本中最新的导出器*

该导出器利用 torch.export  引擎，以提前编译 (AOT) 的方式生成一个仅表示函数中张量计算的追踪图。生成的追踪图 (1) 在功能性 ATen 算子集（以及任何用户指定的自定义算子）中生成标准化的算子，(2) 消除了所有 Python 控制流和数据结构（某些例外情况除外），并且 (3) 在最终转换为 ONNX 图之前，记录了证明这种标准化和控制流消除对于未来输入是可靠所需的一组形状约束。

[了解更多关于基于 torch.export 的 ONNX 导出器 <onnx_export>](了解更多关于基于 torch.export 的 ONNX 导出器 <onnx_export>.md)

## 常见问题

问：我已经导出了我的 LLM 模型，但它的输入尺寸似乎是固定的？

  追踪器会记录示例输入的形状。如果模型应该接受动态形状的输入，请在调用 `torch.onnx.export` 时设置 ``dynamic_shapes``。

问：如何导出包含循环的模型？

  请参阅 torch.cond 。

## 贡献 / 开发

ONNX 导出器是一个社区项目，我们欢迎贡献。我们遵循 [PyTorch 贡献指南](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)，但您可能也有兴趣阅读我们的 [开发维基](https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter)。

## torch.onnx API


### 函数


### 类


### 已弃用的 API
