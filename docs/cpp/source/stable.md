# Torch Stable API

PyTorch Stable C++ API 提供了一个便捷的高级接口，用于调用 ABI 稳定的张量操作以及自定义算子中常用的其他工具。这些函数旨在跨 PyTorch 版本保持二进制兼容性，使其适用于提前编译的代码。

有关稳定 ABI 的更多信息，请参阅 [Stable ABI notes](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)。

## 库注册宏

这些宏提供了标准 PyTorch 算子注册宏（`TORCH_LIBRARY`、`TORCH_LIBRARY_IMPL` 等）的稳定 ABI 等效版本。在构建需要跨 PyTorch 版本保持二进制兼容性的自定义算子时使用这些宏。

### `STABLE_TORCH_LIBRARY(ns, m)`

使用稳定 ABI 在命名空间中定义一个算子库。

这是 `TORCH_LIBRARY`{.interpreted-text role="c:macro"} 的稳定 ABI 等效版本。 使用此宏定义将在跨 PyTorch 版本时保持二进制兼容性的算子模式。每个命名空间只能存在一个 `STABLE_TORCH_LIBRARY` 块；对于来自不同翻译单元的同一命名空间中的额外定义，请使用 `STABLE_TORCH_LIBRARY_FRAGMENT`。

**参数：**

- `ns` - 定义算子的命名空间（例如，`mylib`）。
- `m` - 块中可用的 StableLibrary 变量的名称。

**示例：**

``` cpp
STABLE_TORCH_LIBRARY(mylib, m) {
    m.def("my_op(Tensor input, int size) -> Tensor");
    m.def("another_op(Tensor a, Tensor b) -> Tensor");
}
```

最低兼容版本：PyTorch 2.9。

### `STABLE_TORCH_LIBRARY_IMPL(ns, k, m)`

使用稳定 ABI 为特定调度键注册算子实现。

这是 `TORCH_LIBRARY_IMPL` 的稳定 ABI 等效版本。使用此宏为特定调度键（例如，CPU、CUDA）提供算子实现，同时保持跨 PyTorch 版本的二进制兼容性。

 note
 title
Note


使用此宏注册的所有内核函数必须使用 `TORCH_BOX` 宏进行装箱。


**参数：**

- `ns` - 定义算子的命名空间。
- `k` - 调度键（例如，`CPU`、`CUDA`）。
- `m` - 块中可用的 StableLibrary 变量的名称。

**示例：**

``` cpp
STABLE_TORCH_LIBRARY_IMPL(mylib, CPU, m) {
    m.impl("my_op", TORCH_BOX(&my_cpu_kernel));
}

STABLE_TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("my_op", TORCH_BOX(&my_cuda_kernel));
}
```

最低兼容版本：PyTorch 2.9。

### `STABLE_TORCH_LIBRARY_FRAGMENT(ns, m)`

使用稳定 ABI 扩展现有命名空间中的算子定义。

这是 `TORCH_LIBRARY_FRAGMENT` 的稳定 ABI 等效版本。使用此宏向已使用 `STABLE_TORCH_LIBRARY` 创建的命名空间添加额外的算子定义。

**参数：**

- `ns` - 要扩展的命名空间。
- `m` - 块中可用的 StableLibrary 变量的名称。

最低兼容版本：PyTorch 2.9。

`TORCH_BOX(&func)` \^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^

包装一个函数以符合稳定的装箱内核调用约定。

此宏接受一个未装箱的内核函数指针，并生成一个可以注册到稳定库 API 的装箱包装器。

**参数：**

- `func` - 要包装的未装箱内核函数。

**示例：**

``` cpp
Tensor my_kernel(const Tensor& input, int64_t size) {
    return input.reshape({size});
}

STABLE_TORCH_LIBRARY_IMPL(my_namespace, CPU, m) {
    m.impl("my_op", TORCH_BOX(&my_kernel));
}
```

最低兼容版本：PyTorch 2.9。

## Tensor 类

`torch::stable::Tensor` 类提供了一个用户友好的 C++ 接口，类似于 `torch::Tensor`，同时保持跨 PyTorch 版本的二进制兼容性。

 {.doxygenclass members=""}
torch::stable::Tensor


## Device 类

`torch::stable::Device` 类提供了一个用户友好的 C++ 接口，类似于 `c10::Device`，同时保持跨 PyTorch 版本的二进制兼容性。它表示一个计算设备（CPU、CUDA 等），并带有一个可选的设备索引。

 {.doxygenclass members=""}
torch::stable::Device


## DeviceGuard 类

`torch::stable::accelerator::DeviceGuard` 提供了一个用户友好的 C++ 接口，类似于 `c10::DeviceGuard`，同时保持跨 PyTorch 版本的二进制兼容性。

 {.doxygenclass members=""}
torch::stable::accelerator::DeviceGuard


 doxygenfunction
torch::stable::accelerator::getCurrentDeviceIndex


## 流工具

对于 CUDA 流访问，我们目前推荐使用 ABI 稳定的 C 封装 API。在未来的版本中，将通过一个更符合人体工程学的包装器来改进这一点。

### 获取当前 CUDA 流

要获取当前 `cudaStream_t` 以在 CUDA 内核中使用：

``` cpp
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>

// 目前，我们依赖 ABI 稳定的 C 封装 API 来获取当前 CUDA 流。
// 这将在未来的版本中得到改进。
// 使用 C 封装 API 时，我们需要使用 TORCH_ERROR_CODE_CHECK 来
// 检查错误代码，否则抛出适当的 runtime_error。
void* stream_ptr = nullptr;
TORCH_ERROR_CODE_CHECK(
    aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

// 现在你可以在 CUDA 内核启动中使用 'stream'
my_kernel<<<blocks, threads, 0, stream>>>(args...);
```

 note
 title
Note


使用 C 封装 API 时，必须使用 `TORCH_ERROR_CODE_CHECK` 宏来正确检查错误代码并抛出适当的异常。


## CUDA 错误检查宏

这些宏提供了用于 CUDA 错误检查的稳定 ABI 等效功能。 它们封装了 CUDA API 调用和内核启动，使用 PyTorch 的错误格式化功能提供详细的错误信息。

### `STD_CUDA_CHECK(EXPR)`

检查 CUDA API 调用的结果，并在出错时抛出异常。 使用此宏的用户需要包含 `cuda_runtime.h`。

**示例：**

``` cpp
STD_CUDA_CHECK(cudaMalloc(&ptr, size));
STD_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
```

最低兼容版本：PyTorch 2.10。

### `STD_CUDA_KERNEL_LAUNCH_CHECK()`

检查最近一次 CUDA 内核启动的错误。等同于 `STD_CUDA_CHECK(cudaGetLastError())`。

**示例：**

``` cpp
my_kernel<<<blocks, threads, 0, stream>>>(args...);
STD_CUDA_KERNEL_LAUNCH_CHECK();
```

最低兼容版本：PyTorch 2.10。

## 仅头文件实用工具

`torch::headeronly` 命名空间提供了常见 PyTorch 类型和实用工具的仅头文件版本。 这些工具可以在不链接 libtorch 的情况下使用，使其成为跨 PyTorch 版本维护二进制兼容性的理想选择。

### 错误检查

`STD_TORCH_CHECK` 是一个用于运行时断言的仅头文件宏：

``` cpp
#include <torch/headeronly/util/Exception.h>

STD_TORCH_CHECK(condition, "Error message with ", variable, " interpolation");
```

### 核心类型

以下 `c10::` 类型在 `torch::headeronly::` 命名空间下提供了仅头文件版本：

- `torch::headeronly::ScalarType` - 张量数据类型（Float、Double、Int 等）
- `torch::headeronly::DeviceType` - 设备类型（CPU、CUDA 等）
- `torch::headeronly::MemoryFormat` - 内存布局格式（Contiguous、ChannelsLast 等）
- `torch::headeronly::Layout` - 张量布局（Strided、Sparse 等）

``` cpp
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/Layout.h>

auto dtype = torch::headeronly::ScalarType::Float;
auto device_type = torch::headeronly::DeviceType::CUDA;
auto memory_format = torch::headeronly::MemoryFormat::Contiguous;
auto layout = torch::headeronly::Layout::Strided;
```

### TensorAccessor

`TensorAccessor` 提供了对张量数据的高效、边界检查访问。 你可以从稳定张量的数据指针、大小和步幅构造一个：

``` cpp
#include <torch/headeronly/core/TensorAccessor.h>

// 为 2D 浮点张量创建 TensorAccessor
auto sizes = tensor.sizes();
auto strides = tensor.strides();
torch::headeronly::TensorAccessor<float, 2> accessor(
    static_cast<float*>(tensor.mutable_data_ptr()),
    sizes.data(),
    strides.data());

// 访问元素
float value = accessor[i][j];
```

### 分发宏

仅头文件的分发宏（THO = Torch Header Only）可用于数据类型和设备分发：

``` cpp
#include <torch/headeronly/core/Dispatch.h>

THO_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "my_kernel", [&] {
    // scalar_t 是解析后的类型
    auto* data = tensor.data_ptr<scalar_t>();
});
```

### 完整 API 列表

有关仅头文件 API 的完整列表，请参阅 PyTorch 源代码树中的 `torch/header_only_apis.txt`。

## 稳定运算符

### 张量创建

 doxygenfunction
torch::stable::empty


 doxygenfunction
torch::stable::empty_like


 doxygenfunction
torch::stable::new_empty(const torch::stable::Tensor &self, torch::headeronly::IntHeaderOnlyArrayRef size, std::optional\<torch::headeronly::ScalarType\> dtype, std::optional\<torch::headeronly::Layout\> layout, std::optional\<torch::stable::Device\> device, std::optional\<bool\> pin_memory)


 doxygenfunction
torch::stable::new_zeros(const torch::stable::Tensor &self, torch::headeronly::IntHeaderOnlyArrayRef size, std::optional\<torch::headeronly::ScalarType\> dtype, std::optional\<torch::headeronly::Layout\> layout, std::optional\<torch::stable::Device\> device, std::optional\<bool\> pin_memory)


 doxygenfunction
torch::stable::full


 doxygenfunction
torch::stable::from_blob


### 张量操作

 doxygenfunction
torch::stable::clone


 doxygenfunction
torch::stable::contiguous


 doxygenfunction
torch::stable::reshape


 doxygenfunction
torch::stable::view


 doxygenfunction
torch::stable::flatten


 doxygenfunction
torch::stable::squeeze


 doxygenfunction
torch::stable::unsqueeze


 doxygenfunction
torch::stable::transpose


 doxygenfunction
torch::stable::select


 doxygenfunction
torch::stable::narrow


 doxygenfunction
torch::stable::pad


### 设备和类型转换

 doxygenfunction
torch::stable::to(const torch::stable::Tensor &self, std::optional\<torch::headeronly::ScalarType\> dtype, std::optional\<torch::headeronly::Layout\> layout, std::optional\<torch::stable::Device\> device, std::optional\<bool\> pin_memory, bool non_blocking, bool copy, std::optional\<torch::headeronly::MemoryFormat\> memory_format)


 doxygenfunction
torch::stable::to(const torch::stable::Tensor &self, torch::stable::Device device, bool non_blocking, bool copy)


 doxygenfunction
torch::stable::fill\_


 doxygenfunction
torch::stable::zero\_


 doxygenfunction
torch::stable::copy\_


 doxygenfunction
torch::stable::matmul


 doxygenfunction
torch::stable::amax(const torch::stable::Tensor &self, int64_t dim, bool keepdim)


 doxygenfunction
torch::stable::amax(const torch::stable::Tensor &self, torch::headeronly::IntHeaderOnlyArrayRef dims, bool keepdim)


 doxygenfunction
torch::stable::sum


 doxygenfunction
torch::stable::sum_out


 doxygenfunction
torch::stable::subtract


 doxygenfunction
torch::stable::parallel_for


 doxygenfunction
torch::stable::get_num_threads


### 并行化实用工具

 doxygenfunction
torch::stable::parallel_for


 doxygenfunction
torch::stable::get_num_threads

