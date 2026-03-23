# LibTorch 稳定 ABI

## 概述

LibTorch 稳定 ABI（应用程序二进制接口）提供了一套有限的接口，用于扩展 PyTorch 功能，而无需与特定 PyTorch 版本紧密耦合。这使得开发的自定义算子和扩展能够跨 PyTorch 版本保持兼容。这套有限的 API 集合并非旨在替代现有的 LibTorch，而是为大多数自定义扩展用例提供稳定的基础。如果您希望将任何 API 添加到稳定 ABI 中，请通过 [PyTorch 仓库的新问题](https://github.com/pytorch/pytorch/issues) 提交请求。

有限的稳定 ABI 包含三个主要组成部分：

1. **稳定 C 头文件** - 由 libtorch 实现的底层 C API（主要是 `torch/csrc/inductor/aoti_torch/c/shim.h`）
2. **仅头文件的 C++ 库** - 仅在头文件中实现的独立实用程序，不依赖 libtorch（`torch/headeronly/*`）
3. **稳定 C++ 包装器** - 高级 C++ 便捷包装器（`torch/csrc/stable/*`）

我们将详细讨论这些组件

### `torch/headeronly`

位于 [`torch/headeronly`](https://github.com/pytorch/pytorch/tree/main/torch/headeronly) 的内联 C++ 头文件与 LibTorch 完全解耦。这些头文件包含自定义扩展开发者可能熟悉的某些实用程序。例如，`c10::ScalarType` 枚举在这里作为 `torch::headeronly::ScalarType` 存在，以及独立于 libtorch 的 `TORCH_CHECK` 版本 `STD_TORCH_CHECK`。您可以信任 `torch::headeronly` 命名空间中的所有 API 都不依赖 `libtorch.so`。这些 API 也全局列在 [torch/header_only_apis.txt](https://github.com/pytorch/pytorch/blob/main/torch/header_only_apis.txt) 中。

### `torch/csrc/stable`

这是一组内联 C++ 头文件，为 C API 提供包装器，处理下面讨论的粗糙边缘。

它包括：

- torch/csrc/stable/library.h：提供 TORCH_LIBRARY 及类似宏的稳定版本。
- torch/csrc/stable/tensor_struct.h：提供 torch::stable::Tensor，这是 at::Tensor 的稳定版本。
- torch/csrc/stable/ops.h：提供从 `native_functions.yaml` 调用 ATen 算子的稳定接口。
- torch/csrc/stable/accelerator.h：为设备通用对象和 API 提供稳定接口（例如 `getCurrentStream`、`DeviceGuard`）。

我们正在继续改进 `torch/csrc/stable` API 的覆盖范围。如果您希望在自定义扩展中看到对特定 API 的支持，请提交问题。

有关稳定算子的完整 API 文档，请参阅 [Torch 稳定 API cpp 文档](https://docs.pytorch.org/cppdocs/stable.html)。<!-- @lint-ignore: URL won't exist till stable.rst cpp docs are published in 2.10 -->

### 稳定 C 头文件

由 AOTInductor 启动的稳定 C 头文件构成了稳定 ABI 的基础。目前，可用的 C 头文件包括：

- [torch/csrc/inductor/aoti_torch/c/shim.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/inductor/aoti_torch/c/shim.h)：包含关于张量、数据类型、CUDA 等的常用 C 风格垫片 API。
- [torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h)：包含来自 `native_functions.yaml` 的 ATen 算子的 C 风格垫片 API（例如 `aoti_torch_aten_new_empty`）。
- [torch/csrc/inductor/aoti_torch/generated/c_shim_*.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/inductor/aoti_torch/generated)：包含来自 `native_functions.yaml` 分发的特定后端内核的 C 风格垫片 API（例如 `aoti_torch_cuda_pad`）。这些 API 应仅用于它们命名的特定后端（例如 `aoti_torch_cuda_pad` 应仅在 CUDA 内核中使用），因为它们选择退出分发器。
- [torch/csrc/stable/c/shim.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/stable/c/shim.h)：我们正在构建更多 ABI，逻辑上位于 `torch/csrc/stable/c` 中，而不是继续使用 AOTI 命名，这对我们的通用用例不再有意义。

这些头文件承诺在版本间保持 ABI 稳定，并遵循比 LibTorch 更强的向后兼容性策略。具体来说，我们承诺在发布后至少 2 年内不修改它们。然而，这**风险自负**。例如，用户必须处理某些 API 返回的对象的内存生命周期。此外，下面讨论的基于堆栈的 API 允许用户调用 PyTorch 分发器，但不提供对所调用底层算子的前向和后向兼容性的强保证。

除非绝对必要，我们推荐使用 `torch/csrc/stable` 中的高级 C++ API，它将为用户处理 C API 的所有粗糙边缘。

## 将您的内核迁移到 LibTorch 稳定 ABI

如果您希望您的内核与 LibTorch 保持 ABI 稳定，意味着您能够为一个版本构建并在另一个版本上运行，您的内核必须仅使用有限的稳定 ABI。以下部分将介绍迁移现有内核的一些步骤以及我们想象您需要替换的 API。

首先，LibTorch ABI 稳定内核必须通过 `STABLE_TORCH_LIBRARY` 注册，而不是通过 `TORCH_LIBRARY` 注册。请注意，通过 `STABLE_TORCH_LIBRARY` 注册的实现必须是盒装的，这与 `TORCH_LIBRARY` 不同。`TORCH_BOX` 宏为大多数用例自动处理这一点。请参阅下面的简单示例或我们的 [基于堆栈的 API](stack-based-apis) 文档以获取更多详细信息。对于通过 `pybind` 注册的内核，在使用稳定 ABI 之前，将其迁移到通过 `TORCH_LIBRARY` 注册会很有用。

虽然之前你的内核可能包含来自 `<torch/*.h>` 的 API（例如 `<torch/all.h>`），但现在仅限于包含上述 3 类头文件（`torch/csrc/stable/*.h`、`torch/headeronly/*.h` 和稳定的 C 头文件）。这意味着你的扩展不应再使用 `at::` 或 `c10::` 命名空间中的任何工具，而应使用它们在 `torch::stable` 和 `torch::headeronly` 中的替代品。以下是一些必要的迁移示例：
- 所有 `at::Tensor` 的使用必须替换为 `torch::stable::Tensor`
- 所有 `TORCH_CHECK` 的使用必须替换为 `STD_TORCH_CHECK`
- 所有 `at::kCUDA` 的使用必须替换为 `torch::headeronly::kCUDA` 等
- 诸如 `at::pad` 之类的原生函数必须替换为 `torch::stable::pad`
- 作为 Tensor 方法调用的原生函数（例如 `Tensor.pad`）必须通过 `torch::stable::pad` 替换为 ATen 变体。

如前所述，LibTorch 稳定 ABI 仍在开发中。如果你希望任何 API 或功能被添加到稳定 ABI/`torch::headeronly`/`torch::stable` 中，请通过 [PyTorch 仓库的新 issue](https://github.com/pytorch/pytorch/issues) 提交请求。

以下是一个将使用 `TORCH_LIBRARY` 的现有内核迁移到稳定 ABI（`TORCH_STABLE_LIBRARY`）的简单示例。如需查看更大的端到端示例，可以参考 FA3 仓库。具体来说，可以查看 [`flash_api.cpp`](https://github.com/Dao-AILab/flash-attention/blob/ad70a007e6287d4f7e766f94bcf2f9a813f20f6b/hopper/flash_api.cpp#L1) 与其稳定变体 [`flash_api_stable.cpp`](https://github.com/Dao-AILab/flash-attention/blob/ad70a007e6287d4f7e766f94bcf2f9a813f20f6b/hopper/flash_api_stable.cpp#L1) 之间的差异。


### 使用 `TORCH_LIBRARY` 的原始版本

```cpp
// original_kernel.cpp - Using TORCH_LIBRARY (not stable ABI)
#include <torch/torch.h>
#include <ATen/ATen.h>

namespace myops {

// Simple kernel that adds a scalar value to each element of a tensor
at::Tensor add_scalar(const at::Tensor& input, double scalar) {
  TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");

  return input.add(scalar);
}

// Register the operator
TORCH_LIBRARY(myops, m) {
  m.def("add_scalar(Tensor input, float scalar) -> Tensor");
}

// Register the implementation
TORCH_LIBRARY_IMPL(myops, CompositeExplicitAutograd, m) {
  m.impl("add_scalar", &add_scalar);
}

} // namespace myops
```

### 使用 `STABLE_TORCH_LIBRARY` 的迁移版本

```cpp
// stable_kernel.cpp - Using STABLE_TORCH_LIBRARY (stable ABI)

// (1) Don't include <torch/torch.h> <ATen/ATen.h>
//     only include APIs from torch/csrc/stable, torch/headeronly and C-shims
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

namespace myops {

// Simple kernel that adds a scalar value to each element of a tensor
torch::stable::Tensor add_scalar(const torch::stable::Tensor& input, double scalar) {
  // (2) use STD_TORCH_CHECK instead of TORCH_CHECK
  STD_TORCH_CHECK(
      // (3) use torch::headeronly::kFloat instead of at:kFloat
      input.scalar_type() == torch::headeronly::kFloat,
      "Input must be float32");

  // (4) Use stable ops namespace instead of input.add
  return torch::stable::add(input, scalar);
}

// (5) Register the operator using STABLE_TORCH_LIBRARY
STABLE_TORCH_LIBRARY(myops, m) {
  m.def("add_scalar(Tensor input, float scalar) -> Tensor");
}

// (6) Register the implementation using STABLE_TORCH_LIBRARY_IMPL
//     Use TORCH_BOX to automatically handle boxing/unboxing
STABLE_TORCH_LIBRARY_IMPL(myops, CompositeExplicitAutograd, m) {
  m.impl("add_scalar", TORCH_BOX(&add_scalar));
}

} // namespace myops
```


## 与调度器交互时，对象如何跨 ABI 边界传递？

当通过稳定 API（`STABLE_TORCH_LIBRARY` 等）与调度器交互时，我们使用装箱约定。参数和返回值表示为 `StableIValue` 栈，这与 `torch::jit::stack` 的 IValues 栈相对应。我们将在下面讨论以下内容：
1. StableIValue 转换
2. StableIValue 栈约定
3. 与调度器交互的稳定 API

### StableIValue 转换

我们为用户提供了在 `torch/csrc/stable/stableivalue_conversions.h` 中使用同名的 `to` 和 `from` API 将对象与 StableIValues 相互转换的工具。我们在下面记录了稳定自定义扩展表示、libtorch 表示和 StableIValue 表示。我们完全支持的类型是表中已完成行的类型。你可以依赖这个子集来实现正确的 ABI 稳定性，这意味着你可以在这些类型上调用 `to<T_custom_ext>(arg/ret)` 或 `from(T)`。

对于有限的一组用例，我们还隐式支持任何可在 64 位内表示的字面量类型作为 StableIValues，因为默认的 reinterpret_cast 会成功。（例如：c10::Device。）这些类型目前尽最大努力保持 ABI 稳定，但未来可能会破坏，因此应仅用于短期测试。

即使自定义扩展中没有设备的标准定义表示，你也可以始终在自定义内核中使用 StableIValue 抽象来处理诸如 c10::Device 之类的类型，方法是不深入检查 StableIValue。例如，自定义运算符可以将 StableIValue 设备作为参数，并通过 `aoti_torch_call_dispatcher` 直接将其传递给 aten 运算符。

# LibTorch 稳定 ABI

1. 自定义扩展中的类型：最终用户自定义库中使用的类型。
2. StableIValue 表示：以 ABI 稳定的方式，在用户模型与 libtorch.so 之间进行类型转换的稳定表示。
3. libtorch 中的类型：libtorch.so（或任何与 libtorch 绑定的代码二进制文件）内部使用的类型。
4. 模式类型：模式描述的类型，我们将其视为 native_functions.yaml 中 ATen 操作符以及通过 TORCH_LIBRARY 或 torch.library 注册到分发器的用户自定义操作符的权威来源。

| 自定义扩展中的类型 | StableIValue 表示 | libtorch 中的类型 | 模式类型 |
| -------- | ------- | ------- | ------- |
| std::optional\<S> | 如果有值，则按位原始复制到 uint64_t 指针的前导字节中，指向一个表示 S 的新 StableIValue。如果没有值，则为 nullptr。 | std::optional\<T> | Type? |
| torch::stable::Tensor | 将底层的 AtenTensorHandle 按位原始复制到 uint64_t 的前导字节中 | at::Tensor | Tensor |
| torch::headeronly::ScalarType | 将转换后的底层枚举按位原始复制到 uint64_t 的前导字节中 | torch::headeronly::ScalarType | ScalarType |
| torch::headeronly::Layout | 将转换后的底层枚举按位原始复制到 uint64_t 的前导字节中 | at::Layout | Layout |
| torch::headeronly::MemoryFormat | 将转换后的底层枚举按位原始复制到 uint64_t 的前导字节中 | at::MemoryFormat | MemoryFormat |
| bool | 按位原始复制到 uint64_t 的前导字节中 | bool | bool |
| int64_t | 按位原始复制到 uint64_t 的前导字节中 | int64_t | int |
| double | 按位原始复制到 uint64_t 的前导字节中 | double | float |
| torch::stable::Device | 将索引和类型按位原始复制到 uint64_t 的前导字节中 | c10::Device | Device |
| ? | ? | c10::Stream | Stream |
| ? | ? | c10::complex<double> | complex |
| ? | ? | at::Scalar | Scalar |
| std::string/std::string_view | 将底层的 StringHandle 按位原始复制到 uint64_t 的前导字节中 | std::string/const char*/ivalue::ConstantString | str |
| ? | ? | at::Storage | Storage |
| ? | ? | at::Generator | Generator |
| std::vector<T>/torch::headeronly::HeaderOnlyArrayRef<T> | 按位原始复制到 uint64_t 指针的前导字节中，指向一个新的 StableIValue，该 StableIValue 指向一个列表，列表中的 StableIValue 递归地表示底层元素。 | c10::List\<T> | Type[] |
| ? | ? | ivalue::Tuple\<T> | (Type, ...) |
| ? | ? | c10::SymInt | SymInt |
| ? | ? | c10::SymFloat | SymFloat |
| ? | ? | c10::SymBool | SymBool |
| ? | ? | at::QScheme | QScheme |

### 栈约定

栈有两个不变式：

1. 栈从左到右填充。
    a. 例如，表示参数 `arg0`、`arg1` 和 `arg2` 的栈将在索引 0 处存放 `arg0`，索引 1 处存放 `arg1`，索引 2 处存放 `arg2`。
    b. 返回值也按从左到右的顺序填充，例如，`ret0` 将在索引 0 处，`ret1` 将在索引 1 处，依此类推。

2. 栈始终拥有其持有的对象的所有权。
    a. 当调用基于栈的 API 时，必须向调用栈提供拥有所有权的引用，并从返回的栈中窃取引用。
    b. 当注册函数以通过栈调用时，必须从参数栈中窃取引用，并将新的引用推入栈中。

(stack-based-apis)=
### 基于栈的 API

上述内容在两个地方相关：

1. `STABLE_TORCH_LIBRARY`
    与 `TORCH_LIBRARY` 不同，分发器期望通过 `STABLE_TORCH_LIBRARY` 注册的内核是盒装的。`TORCH_BOX` 宏会自动为您处理此盒装操作：

    ```cpp
    Tensor my_amax_vec(Tensor t) {
        std::vector<int64_t> v = {0,1};
        return amax(t, v, false);
    }

    // 使用 TORCH_BOX 自动生成盒装包装器
    STABLE_TORCH_LIBRARY(myops, m) {
        m.def("my_amax_vec(Tensor t) -> Tensor", TORCH_BOX(&my_amax_vec));
    }
    ```

2. `torch_call_dispatcher`
    此 API 允许您从 C/C++ 代码调用 PyTorch 分发器。它具有以下签名：

    ```cpp
    torch_call_dispatcher(const char* opName, const char* overloadName, StableIValue* stack, uint64_t extension_build_version);
    ```

    `torch_call_dispatcher` 将调用由给定的 `opName`、`overloadName`、StableIValue 栈以及用户扩展的 `TORCH_ABI_VERSION` 定义的操作符重载。此调用将操作符的任何返回值以 StableIValue 形式填充到栈中，`ret0` 在索引 0 处，`ret1` 在索引 1 处，依此类推。

    我们建议不要使用此 API 调用由其他扩展注册到分发器的函数，除非调用者能够保证他们期望的签名与自定义扩展注册的签名匹配。

### 版本控制与前向/后向兼容性保证

我们在 `torch/headeronly/version.h` 中提供了一个 `TORCH_ABI_VERSION` 宏，其形式如下：

```
[ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ]
[主版本][次版本][补丁版本][                 ABI 标签              ]
```

在当前的开发阶段，C-shim 中的 API 将根据它们首次引入的主版本.次版本.补丁版本进行版本控制，2.10 将是强制执行此规则的首个版本。ABI 标签保留供将来使用。

扩展可以选择要兼容的最低 ABI 版本，方法是在包含任何稳定头文件之前使用：

```
#define TORCH_TARGET_VERSION (((0ULL + major) << 56) | ((0ULL + minor) << 48))
```

或者通过向编译器传递等效的 `-D` 选项。否则，默认值将是当前的 `TORCH_ABI_VERSION`。

以上机制确保，如果用户将 `TORCH_TARGET_VERSION` 定义为 0x0209000000000000（2.9）并尝试使用在版本 2.10 中引入的 C 接口 API `foo`，将会引发编译错误。类似地，`torch/csrc/stable` 中的 C++ 包装器 API 与旧版 libtorch 二进制文件兼容，兼容性上限为其被引入时的 TORCH_ABI_VERSION，并且向前兼容更新版本的 libtorch 二进制文件。

`torch/csrc/stable` 或 `torch/headeronly` 中的 C++ API 遵循与 PyTorch 其他部分相同的向前/向后兼容性策略（参见[策略](https://github.com/pytorch/pytorch/wiki/PyTorch's-Python-Frontend-Backward-and-Forward-Compatibility-Policy)）。LibTorch ABI 稳定的 C 接口 API 保证至少有两年的兼容性窗口期。