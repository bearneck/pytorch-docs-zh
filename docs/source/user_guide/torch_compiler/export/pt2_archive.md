
# PT2 归档规范

以下规范定义了可通过以下方法生成的归档格式：

* 通过调用 `torch.export.save` 的 `torch.export <torch.export>`
* 通过调用 `torch._inductor.aoti_compile_and_package` 的 `AOTInductor <torch.compiler_aot_inductor>`

该归档是一个 zip 文件，可以使用标准的 zipfile API 进行操作。

以下是一个示例归档。我们将按文件夹逐一介绍归档内容。

```
.
├── archive_format
├── byteorder
├── .data
│   ├── serialization_id
│   └── version
├── data
│   ├── aotinductor
│   │   └── model1
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.kernel_metadata.json
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.kernel.cpp
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.wrapper_metadata.json
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.wrapper.cpp
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.wrapper.so
│   │       ├── cg7domx3woam3nnliwud7yvtcencqctxkvvcafuriladwxw4nfiv.cubin
│   │       └── cubaaxppb6xmuqdm4bej55h2pftbce3bjyyvljxbtdfuolmv45ex.cubin
│   ├── weights
│   │  ├── model1_weights_config.json
│   │  ├── model2_weights_config.json
│   │  ├── weight_0
│   │  ├── weight_1
│   │  ├── weight_2
│   └── constants
│   │  ├── model1_constants_config.json
│   │  ├── model2_constants_config.json
│   │  ├── tensor_0
│   │  ├── tensor_1
│   │  ├── custom_obj_0
│   │  ├── custom_obj_1
│   └── sample_inputs
│       ├── model1.pt
│       └── model2.pt
├── extra
│   └── ....json
└── models
    ├── model1.json
    └── model2.json
```

## 内容

### 归档头部

* `archive_format` 声明此归档使用的格式。目前，它只能是 "pt2"。
* `byteorder`。取值为 "little" 或 "big" 之一，供 zip 文件读取器使用。
* `/.data/version` 包含归档版本。（请注意，这既不是导出序列化的模式版本，也不是 Aten 操作集版本）。
* `/.data/serialization_id` 是为当前归档生成的哈希值，用于验证。

### AOTInductor 编译产物

路径：`/data/aotinductor/<model_name>-<backend>/`

AOTInductor 编译产物会为每个模型-后端对保存。例如，`model1` 模型在 A100 和 H100 上的编译产物将分别保存在 `model1-a100` 和 `model1-h100` 文件夹中。

该文件夹通常包含：
* `<uuid>.wrapper.so`：由 `<uuid>.cpp` 编译的动态库。
* `<uuid>.wrapper.cpp`：AOTInductor 生成的 cpp 包装器文件。
* `<uuid>.kernel.cpp`：AOTInductor 生成的 cpp 内核文件。
* `*.cubin`：由 Triton 代码生成内核编译得到的 Triton 内核。
* `<uuid>.wrapper_metadata.json`：从 `aot_inductor.metadata` 电感器配置传入的元数据。
* （可选）`<uuid>.json`：供 `ProxyExecutor` 执行的自定义操作的外部回退节点，根据 `ExternKernelNode` 结构序列化。如果模型不使用自定义操作/ProxyExecutor，此文件将被省略。

### 权重

路径：`/data/weights/*`

模型参数和缓冲区保存在 `/data/weights/` 文件夹中。每个张量保存为一个单独的文件。该文件仅包含原始数据块，张量元数据以及从模型权重 FQN 到保存的原始数据块的映射分别保存在 `<model_name>_weights_config.json` 中。

### 常量

路径：`/data/constants/*`

TensorConstants、非持久缓冲区和 TorchBind 对象保存在 `/data/constants/` 文件夹中。元数据以及从模型常量 FQN 到保存的原始数据块的映射分别保存在 `<model_name>_constants_config.json` 中。

### 示例输入

路径：`/data/sample_inputs/<model_name>.pt`

`torch.export` 使用的 `sample_input` 可以包含在归档中供下游使用。通常，它是一个扁平化的张量列表，结合了 forward() 函数的 args 和 kwargs。

.pt 文件由 `torch.save(sample_input)` 生成，可以在 Python 中通过 `torch.load()` 和在 C++ 中通过 `torch::pickle_load()` 加载。

当模型有多个示例输入副本时，它们将被打包为 `<model_name>_<index>.pt`。

### 模型定义

路径：`/models/<model_name>.json`

模型定义是来自 `torch.export.save` 的 ExportedProgram 的序列化 json，以及其他模型级别的元数据。

## 多模型

此归档规范支持多个模型定义共存于同一个文件中，`<model_name>` 作为模型的唯一标识符，并将在归档的其他文件夹中用作引用。

像 `torch.export.pt2_archive._package.package_pt2` 和 `torch.export.pt2_archive._package.load_pt2` 这样的底层 API 允许您对打包和加载过程进行更细粒度的控制。