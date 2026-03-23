# AOTInductor: Torch.Export 模型的提前编译

AOTInductor 是 [TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) 的一个专门版本，旨在处理导出的 PyTorch 模型，对其进行优化，并生成共享库以及其他相关产物。这些编译后的产物专门设计用于在非 Python 环境中部署，通常用于服务器端的推理部署。

在本教程中，您将了解如何获取一个 PyTorch 模型，将其导出、编译成产物，并使用 C++ 进行模型预测。

## 模型编译

要使用 AOTInductor 编译模型，我们首先需要使用 `torch.export.export` 将给定的 PyTorch 模型捕获为计算图。torch.export  提供了正确性保证以及对捕获的 IR 的严格规范，AOTInductor 依赖于此。

然后，我们将使用 `torch._inductor.aoti_compile_and_package` 通过 TorchInductor 编译导出的程序，并将编译后的产物保存到一个包中。该包采用 PT2 归档规范  的格式。

```{note}
如果您的机器上启用了 CUDA 设备，并且您安装了支持 CUDA 的 PyTorch，以下代码会将模型编译为用于 CUDA 执行的共享库。否则，编译后的产物将在 CPU 上运行。为了在 CPU 推理期间获得更好的性能，建议在运行下面的 Python 脚本之前，通过设置 `export TORCHINDUCTOR_FREEZING=1` 来启用冻结。在配备 Intel® GPU 的环境中，此行为同样适用。
python
import os
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device=device)
    example_inputs=(torch.randn(8, 10, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    # [可选] 将输入 x 的第一个维度指定为动态维度。
    exported = torch.export.export(model, example_inputs, dynamic_shapes={"x": {0: batch_dim}})
    # [注意] 在此示例中，我们直接将导出的模块传递给 aoti_compile_and_package。
    # 根据您的使用情况，例如，如果您的训练平台和推理平台不同，您可以选择使用 torch.export.save 保存导出的模型，
    # 然后在推理平台上使用 torch.export.load 加载回来以运行 AOT 编译。
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        # [可选] 指定生成的共享库路径。如果未指定，
        # 生成的产物将存储在系统的临时目录中。
        package_path=os.path.join(os.getcwd(), "model.pt2"),
        # [可选] 指定 Inductor 配置
        # 这个特定的 max_autotune 选项将开启更广泛的内核自动调优以获得更好的性能。
        inductor_configs={"max_autotune": True,},
    )
```

在这个示例中，`Dim` 参数用于将输入变量 "x" 的第一个维度指定为动态维度。值得注意的是，编译库的路径和名称未指定，导致共享库存储在临时目录中。为了从 C++ 端访问此路径，我们将其保存到文件中，以便稍后在 C++ 代码中检索。

## 在 Python 中进行推理

有多种方法可以部署编译后的产物进行推理，其中一种方法是使用 Python。我们在 Python 中提供了一个便捷的实用程序 API `torch._inductor.aoti_load_package` 用于加载和运行产物，如下例所示：

```python
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))
print(model(torch.randn(8, 10, device=device)))
```

推理时的输入应具有与导出时输入相同的大小、数据类型和步长。

## 在 C++ 中进行推理

接下来，我们使用以下示例 C++ 文件 `inference.cpp` 来加载编译后的产物，使我们能够在 C++ 环境中直接进行模型预测。

```cpp
#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

int main() {
    c10::InferenceMode mode;

    torch::inductor::AOTIModelPackageLoader loader("model.pt2");
    // 假设在 CUDA 上运行
    std::vector<torch::Tensor> inputs = {torch::randn({8, 10}, at::kCUDA)};
    std::vector<torch::Tensor> outputs = loader.run(inputs);
    std::cout << "第一次推理的结果:"<< std::endl;
    std::cout << outputs[0] << std::endl;

    // 第二次推理使用不同的批次大小，这是可行的，因为我们在编译 model.pt2 时将该维度指定为动态维度。
    std::cout << "第二次推理的结果:"<< std::endl;
    // 假设在 CUDA 上运行
    std::cout << loader.run({torch::randn({1, 10}, at::kCUDA)})[0] << std::endl;

    return 0;
}
```

为了构建 C++ 文件，您可以使用提供的 `CMakeLists.txt` 文件，该文件自动化了调用 `python model.py` 进行模型的 AOT 编译以及将 `inference.cpp` 编译成名为 `aoti_example` 的可执行二进制文件的过程。

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aoti_example)

find_package(Torch REQUIRED)

add_executable(aoti_example inference.cpp model.pt2)
```

add_custom_command(
    OUTPUT model.pt2
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/model.py
    DEPENDS model.py
)

target_link_libraries(aoti_example "${TORCH_LIBRARIES}")
set_property(TARGET aoti_example PROPERTY CXX_STANDARD 17)
```

假设目录结构如下所示，您可以执行后续命令来构建二进制文件。需要注意的是，`CMAKE_PREFIX_PATH` 变量对于 CMake 定位 LibTorch 库至关重要，应将其设置为绝对路径。请注意，您的路径可能与本示例中所示的不同。

```
aoti_example/
    CMakeLists.txt
    inference.cpp
    model.py
bash
$ mkdir build
$ cd build
$ CMAKE_PREFIX_PATH=/path/to/python/install/site-packages/torch/share/cmake cmake ..
$ cmake --build . --config Release
```

在 `build` 目录中生成 `aoti_example` 二进制文件后，执行它将显示类似于以下的结果：

```bash
$ ./aoti_example
第一次推理的结果：
0.4866
0.5184
0.4462
0.4611
0.4744
0.4811
0.4938
0.4193
[ CUDAFloatType{8,1} ]
第二次推理的结果：
0.4883
0.4703
[ CUDAFloatType{2,1} ]
```

## 故障排除

以下是一些用于调试 AOT Inductor 的有用工具。

- [/ /logging](../../logging.md)
- [Torch Compiler Aot Inductor Minifier](torch.compiler_aot_inductor_minifier.md)
- [Torch Compiler Aot Inductor Debugging Guide](torch.compiler_aot_inductor_debugging_guide.md)


要启用对输入的运行时检查，请将环境变量 `AOTI_RUNTIME_CHECK_INPUTS` 设置为 1。如果编译模型的输入在大小、数据类型或步幅方面与导出期间使用的输入不同，这将引发 `RuntimeError`。

## API 参考
