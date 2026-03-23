# 基于 torch.export 的 ONNX 导出器


```{contents}
:local:
:depth: 1
```

## 概述

`torch.export <torch.export>` 引擎被用来以提前编译（AOT）的方式生成一个仅表示函数中张量计算过程的追踪图。生成的追踪图（1）在功能性 ATen 算子集（以及任何用户指定的自定义算子）中生成标准化的算子，（2）消除了所有 Python 控制流和数据结构（某些例外情况除外），并且（3）在最终转换为 ONNX 图之前，记录了证明这种标准化和控制流消除对于未来输入是可靠所需的所有形状约束。

此外，在导出过程中，内存使用量显著减少。

## 依赖项

ONNX 导出器依赖于额外的 Python 包：

  - [ONNX](https://onnx.ai)
  - [ONNX Script](https://microsoft.github.io/onnxscript)

可以通过 [pip](https://pypi.org/project/pip/) 安装：

```{code-block} bash

  pip install --upgrade onnx onnxscript
```

然后可以使用 [onnxruntime](https://onnxruntime.ai) 在各种处理器上执行模型。

## 一个简单的示例

以下演示了以简单的多层感知机（MLP）为例的导出器 API 使用：

```{code-block} python
import torch
import torch.nn as nn

class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = nn.Linear(8, 8, bias=True)
      self.fc1 = nn.Linear(8, 4, bias=True)
      self.fc2 = nn.Linear(4, 2, bias=True)
      self.fc3 = nn.Linear(2, 2, bias=True)
      self.fc_combined = nn.Linear(8 + 8 + 8, 8, bias=True)  # 合并所有输入

  def forward(self, tensor_x: torch.Tensor, input_dict: dict, input_list: list):
      """
      需要所有输入的前向方法：
      - tensor_x: 一个直接的张量输入。
      - input_dict: 一个包含键 'tensor_x' 下张量的字典。
      - input_list: 一个列表，其中第一个元素是张量。
      """
      # 从输入中提取张量
      dict_tensor = input_dict['tensor_x']
      list_tensor = input_list[0]

      # 将所有输入合并为一个张量
      combined_tensor = torch.cat([tensor_x, dict_tensor, list_tensor], dim=1)

      # 通过各层处理合并后的张量
      combined_tensor = self.fc_combined(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      combined_tensor = self.fc0(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      combined_tensor = self.fc1(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      combined_tensor = self.fc2(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      output = self.fc3(combined_tensor)
      return output

model = MLPModel()

# 示例输入
tensor_input = torch.rand((97, 8), dtype=torch.float32)
dict_input = {'tensor_x': torch.rand((97, 8), dtype=torch.float32)}
list_input = [torch.rand((97, 8), dtype=torch.float32)]

# input_names 和 output_names 用于标识 ONNX 模型的输入和输出
input_names = ['tensor_input', 'tensor_x', 'list_input_index_0']
output_names = ['output']

# 使用所有必需的输入导出模型
onnx_program = torch.onnx.export(model,(tensor_input, dict_input, list_input), dynamic_shapes=({0: "batch_size"},{"tensor_x": {0: "batch_size"}},[{0: "batch_size"}]), input_names=input_names, output_names=output_names, dynamo=True,)

# 检查导出的 ONNX 模型是否为动态的
assert onnx_program.model.graph.inputs[0].shape == ("batch_size", 8)
assert onnx_program.model.graph.inputs[1].shape == ("batch_size", 8)
assert onnx_program.model.graph.inputs[2].shape == ("batch_size", 8)
```

如上代码所示，您只需要向 `torch.onnx.export` 提供模型实例及其输入。导出器将返回一个 `torch.onnx.ONNXProgram` 实例，其中包含导出的 ONNX 图以及额外信息。

通过 ``onnx_program.model_proto`` 可用的内存中模型是一个符合 [ONNX IR 规范](https://github.com/onnx/onnx/blob/main/docs/IR.md) 的 ``onnx.ModelProto`` 对象。然后可以使用 `torch.onnx.ONNXProgram.save` API 将 ONNX 模型序列化为 [Protobuf 文件](https://protobuf.dev/)。

```{code-block} python
  onnx_program.save("mlp.onnx")
```

## 使用 GUI 检查 ONNX 模型

您可以使用 [Netron](https://netron.app/) 查看导出的模型。

```{image} _static/img/onnx/onnx_dynamo_mlp_model.png
:alt: 使用 Netron 查看的 MLP 模型
:width: 30%
:align: center
```

## 当转换失败时

应使用参数 ``report=True`` 第二次调用函数 `torch.onnx.export`。将生成一个 markdown 报告以帮助用户解决问题。

## 元数据

在 ONNX 导出期间，每个 ONNX 节点都会被标注元数据，这些元数据有助于追踪其在原始 PyTorch 模型中的来源和上下文。此元数据对于调试、模型检查以及理解 PyTorch 和 ONNX 图之间的映射非常有用。

以下元数据字段被添加到每个 ONNX 节点：

- **namespace**

  表示节点层次化命名空间的字符串，由模块/方法的堆栈跟踪组成。

  *示例：*
  `__main__.SimpleAddModel/add: aten.add.Tensor`

- **pkg.torch.onnx.class_hierarchy**

  表示导致此节点的模块层次结构的类名列表。

  *示例：*
  `['__main__.SimpleAddModel', 'aten.add.Tensor']`

- **pkg.torch.onnx.fx_node**

  原始 FX 节点的字符串表示，包括其名称、消费者数量、目标 torch 操作、参数和关键字参数。

  *示例：*
  `%cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tensor_x, %input_dict_tensor_x, %input_list_0], 1), kwargs = {})`

- **pkg.torch.onnx.name_scopes**

  表示此节点在 PyTorch 模型中路径的名称作用域（方法）列表。

  *示例：*
  `['', 'add']`

- **pkg.torch.onnx.stack_trace**

  创建此节点的原始代码中的堆栈跟踪（如果可用）。

  *示例：*
  ```
  File "simpleadd.py", line 7, in forward
      return torch.add(x, y)
  ```

这些元数据字段存储在每个 ONNX 节点的 `metadata_props` 属性中，可以使用 Netron 或通过编程方式进行检查。

整个 ONNX 图具有以下 `metadata_props`：

- **pkg.torch.export.ExportedProgram.graph_signature**

  此属性包含原始 PyTorch ExportedProgram 中 `graph_signature` 的字符串表示形式。图签名描述了模型输入和输出的结构以及它们如何映射到 ONNX 图。输入被定义为 `InputSpec` 对象，其中包括输入的种类（例如，参数为 `InputKind.PARAMETER`，用户定义的输入为 `InputKind.USER_INPUT`）、参数名称、目标（可以是模型中的特定节点）以及输入是否是持久性的。输出被定义为 `OutputSpec` 对象，指定输出的种类（例如，`OutputKind.USER_OUTPUT`）和参数名称。

  要了解更多关于图签名的信息，请参阅 `torch.export <user_guide/torch_compiler/export>` 获取更多信息。

- **pkg.torch.export.ExportedProgram.range_constraints**

  此属性包含原始 PyTorch ExportedProgram 中存在的任何范围约束的字符串表示形式。范围约束指定了模型中符号形状或值的有效范围，这对于使用动态形状或符号维度的模型可能很重要。

  *示例：*
  `s0: VR[2, int_oo]`，表示输入张量的大小必须至少为 2。

  要了解更多关于范围约束的信息，请参阅 `torch.export <user_guide/torch_compiler/export>` 获取更多信息。

ONNX 图中的每个输入值可能具有以下元数据属性：

- **pkg.torch.export.graph_signature.InputSpec.kind**

  输入的种类，由 PyTorch 的 InputKind 枚举定义。

  *示例值：*
  - "USER_INPUT"：用户提供给模型的输入。
  - "PARAMETER"：模型参数（例如，权重）。
  - "BUFFER"：模型缓冲区（例如，BatchNorm 中的运行均值）。
  - "CONSTANT_TENSOR"：常量张量参数。
  - "CUSTOM_OBJ"：自定义对象输入。
  - "TOKEN"：令牌输入。

- **pkg.torch.export.graph_signature.InputSpec.persistent**

  指示输入是否是持久性的（即，应作为模型状态的一部分保存）。

  *示例值：*
  - "True"
  - "False"

ONNX 图中的每个输出值可能具有以下元数据属性：

- **pkg.torch.export.graph_signature.OutputSpec.kind**

  输出的种类，由 PyTorch 的 OutputKind 枚举定义。

  *示例值：*
  - "USER_OUTPUT"：用户可见的输出。
  - "LOSS_OUTPUT"：损失值输出。
  - "BUFFER_MUTATION"：指示缓冲区已发生突变。
  - "GRADIENT_TO_PARAMETER"：参数的梯度输出。
  - "GRADIENT_TO_USER_INPUT"：用户输入的梯度输出。
  - "USER_INPUT_MUTATION"：指示用户输入已发生突变。
  - "TOKEN"：令牌输出。

每个初始化的值、输入、输出具有以下元数据：

- **pkg.torch.onnx.original_node_name**

  在值被重命名的情况下，在 PyTorch FX 图中产生此值的节点的原始名称。这有助于将初始化器追溯回原始模型中的来源。

  *示例：*
  `fc1.weight`

## API 参考

