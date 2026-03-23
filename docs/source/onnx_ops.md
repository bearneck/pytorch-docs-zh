# torch.onnx.ops

```{eval-rst}
.. automodule:: torch.onnx.ops
```

## 符号运算符

这些运算符可用于在 FX 计算图中符号化地创建任何 ONNX 算子。
这些运算符不执行实际计算。建议在 ``if torch.onnx.is_in_onnx_export`` 代码块内使用它们。

```{eval-rst}
.. autofunction:: torch.onnx.ops.symbolic
.. autofunction:: torch.onnx.ops.symbolic_multi_out
```

## ONNX 运算符

以下运算符作为原生 PyTorch 算子实现，可以导出为 ONNX 运算符。它们可以在 ``nn.Module`` 中原生使用。

例如，您可以定义一个模块：

```py
class Model(torch.nn.Module):
    def forward(
        self, input_data, cos_cache_data, sin_cache_data, position_ids_data
    ):
        return torch.onnx.ops.rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            position_ids_data,
        )
```

并使用以下方式将其导出到 ONNX：

```py
input_data = torch.rand(2, 3, 4, 8)
position_ids_data = torch.randint(0, 50, (2, 3)).long()
sin_cache_data = torch.rand(50, 4)
cos_cache_data = torch.rand(50, 4)
dynamic_shapes = {
    "input_data": {0: torch.export.Dim.DYNAMIC},
    "cos_cache_data": None,
    "sin_cache_data": None,
    "position_ids_data": {0: torch.export.Dim.DYNAMIC},
}
onnx_program = torch.onnx.export(
    model,
    (input_data, cos_cache_data, sin_cache_data, position_ids_data),
    dynamic_shapes=dynamic_shapes,
    dynamo=True,
    opset_version=23,
)
```

打印 ONNX 程序将显示图中使用的 ONNX 运算符：

```
<...>

graph(
    name=main_graph,
    inputs=(
        %"input_data"<FLOAT,[s0,3,4,8]>,
        %"cos_cache_data"<FLOAT,[50,4]>,
        %"sin_cache_data"<FLOAT,[50,4]>,
        %"position_ids_data"<INT64,[s0,3]>
    ),
    outputs=(
        %"rotary_embedding"<FLOAT,[s0,3,4,8]>
    ),
) {
    0 |  # rotary_embedding
        %"rotary_embedding"<FLOAT,[s0,3,4,8]> ⬅️ ::RotaryEmbedding(%"input_data", %"cos_cache_data", %"sin_cache_data", %"position_ids_data")
    return %"rotary_embedding"<FLOAT,[s0,3,4,8]>
}
```

对应的 ``ExportedProgram`` 如下：

ExportedProgram:

```py
class GraphModule(torch.nn.Module):
    def forward(self, input_data: "f32[s0, 3, 4, 8]", cos_cache_data: "f32[50, 4]", sin_cache_data: "f32[50, 4]", position_ids_data: "i64[s0, 3]"):
        rotary_embedding: "f32[s0, 3, 4, 8]" = torch.ops.onnx.RotaryEmbedding.opset23(input_data, cos_cache_data, sin_cache_data, position_ids_data);  input_data = cos_cache_data = sin_cache_data = position_ids_data = None
        return (rotary_embedding,)
```

```{eval-rst}
.. autofunction:: torch.onnx.ops.rotary_embedding
.. autofunction:: torch.onnx.ops.attention
```

## ONNX 到 ATen 的分解表

您可以使用 {func}`torch.onnx.ops.aten_decompositions` 来获取一个分解表，将上述定义的 ONNX 运算符分解为 ATen 运算符。

```py
class Model(torch.nn.Module):
    def forward(
        self, input_data, cos_cache_data, sin_cache_data, position_ids_data
    ):
        return torch.onnx.ops.rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            position_ids_data,
        )

model = Model()

ep = torch.export.export(
    model,
    (input_data, cos_cache_data, sin_cache_data, position_ids_data),
)
# 该程序可以分解为 aten 算子
ep_decomposed = ep.run_decompositions(torch.onnx.ops.aten_decompositions())
```

```{eval-rst}
.. autofunction:: torch.onnx.ops.aten_decompositions
```