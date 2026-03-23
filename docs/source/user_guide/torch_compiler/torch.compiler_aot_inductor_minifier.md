# AOTInductor 最小化工具

如果在使用 AOT Inductor API（例如 `torch._inductor.aoti_compile_and_package`、`torch._indcutor.aoti_load_package`）或在某些输入上运行 `aoti_load_package` 加载的模型时遇到错误，您可以通过设置 `from torch._inductor import config; config.aot_inductor.dump_aoti_minifier = True` 来使用 AOTInductor 最小化工具创建一个能复现错误的最小化 nn.Module。

从高层次来看，使用最小化工具有两个步骤：

- 设置 `from torch._inductor import config; config.aot_inductor.dump_aoti_minifier = True` 或设置环境变量 `DUMP_AOTI_MINIFIER=1`。然后运行出错的脚本将生成一个 `minifier_launcher.py` 脚本。输出目录可通过将 `torch._dynamo.config.debug_dir_root` 设置为有效的目录名来配置。

- 运行 `minifier_launcher.py` 脚本。如果最小化工具运行成功，它将在 `repro.py` 中生成可运行的 Python 代码，该代码能复现完全相同的错误。

## 示例代码

以下示例代码将生成一个错误，因为我们通过 `torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = "compile_error"` 在 relu 上注入了一个错误。

```
import torch
from torch._inductor import config as inductor_config

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x


inductor_config.aot_inductor.dump_aoti_minifier = True
torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = "compile_error"

with torch.no_grad():
    model = Model().to("cuda")
    example_inputs = (torch.randn(8, 10).to("cuda"),)
    ep = torch.export.export(model, example_inputs)
    package_path = torch._inductor.aoti_compile_and_package(ep)
    compiled_model = torch._inductor.aoti_load_package(package_path)
    result = compiled_model(*example_inputs)
```

上面的代码生成了以下错误：

```text
RuntimeError: Failed to import /tmp/torchinductor_shangdiy/fr/cfrlf4smkwe4lub4i4cahkrb3qiczhf7hliqqwpewbw3aplj5g3s.py
SyntaxError: invalid syntax (cfrlf4smkwe4lub4i4cahkrb3qiczhf7hliqqwpewbw3aplj5g3s.py, line 29)
```

这是因为我们在 relu 上注入了一个错误，因此生成的 triton 内核如下所示。请注意，我们使用了 `compile error!` 而不是 `relu`，因此我们得到了一个 `SyntaxError`。

```
@triton.jit
def triton_poi_fused_addmm_relu_sigmoid_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = compile error!
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
```

由于我们设置了 `torch._inductor.config.aot_inductor.dump_aoti_minifier=True`，我们还会看到一行额外的信息，指示 `minifier_launcher.py` 被写入的位置。输出目录可通过将 `torch._dynamo.config.debug_dir_root` 设置为有效的目录名来配置。

```text
W1031 16:21:08.612000 2861654 pytorch/torch/_dynamo/debug_utils.py:279] Writing minified repro to:
W1031 16:21:08.612000 2861654 pytorch/torch/_dynamo/debug_utils.py:279] /data/users/shangdiy/pytorch/torch_compile_debug/run_2024_10_31_16_21_08_602433-pid_2861654/minifier/minifier_launcher.py
```

## 最小化工具启动器

`minifier_launcher.py` 文件包含以下代码。`exported_program` 包含传递给 `torch._inductor.aoti_compile_and_package` 的输入。参数 `command='minify'` 意味着脚本将运行最小化工具以创建一个能复现错误的最小化图模块。或者，您可以使用 `command='run'` 来仅编译、加载并运行加载的模型（而不运行最小化工具）。

```
import torch
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = 'compile_error'
torch._inductor.config.aot_inductor.dump_aoti_minifier = True


isolate_fails_code_str = None


# torch version: 2.6.0a0+gitcd9c6e9
# torch cuda version: 12.0
# torch git version: cd9c6e9408dd79175712223895eed36dbdc84f84


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Fri_Jan__6_16:45:21_PST_2023
# Cuda compilation tools, release 12.0, V12.0.140
# Build cuda_12.0.r12.0/compiler.32267302_0

# GPU Hardware Info:
# NVIDIA PG509-210 : 8

exported_program = torch.export.load('/data/users/shangdiy/pytorch/torch_compile_debug/run_2024_11_06_13_52_35_711642-pid_3567062/minifier/checkpoints/exported_program.pt2')
# print(exported_program.graph)
config_patches={}
if __name__ == '__main__':
    from torch._dynamo.repro.aoti import run_repro
    with torch.no_grad():
        run_repro(exported_program, config_patches=config_patches, accuracy=False, command='minify', save_dir='/data/users/shangdiy/pytorch/torch_compile_debug/run_2024_11_06_13_52_35_711642-pid_3567062/minifier/checkpoints', check_str=None)
```

假设我们保留了 `command='minify'` 选项并运行脚本，我们将得到以下输出：

```text
...
W1031 16:48:08.938000 3598491 torch/_dynamo/repro/aoti.py:89] Writing checkpoint with 3 nodes to /data/users/shangdiy/pytorch/torch_compile_debug/run_2024_10_31_16_48_02_720863-pid_3598491/minifier/checkpoints/3.py
W1031 16:48:08.975000 3598491 torch/_dynamo/repro/aoti.py:101] Copying repro file for convenience to /data/users/shangdiy/pytorch/repro.py
Wrote minimal repro out to repro.py
```

如果在运行 `minifier_launcher.py` 时遇到 `AOTIMinifierError`，请在此处提交错误报告。

## 最小化结果

`repro.py` 文件内容如下。请注意，导出的程序会打印在文件顶部，并且它只包含 relu 节点。最小化工具成功地将图缩减到引发错误的算子。

```
# from torch.nn import *
# class Repro(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()


#     def forward(self, linear):
#         relu = torch.ops.aten.relu.default(linear);  linear = None
#         return (relu,)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.generate_intermediate_hooks = True
torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = 'compile_error'
torch._inductor.config.aot_inductor.dump_aoti_minifier = True


isolate_fails_code_str = None


# torch version: 2.6.0a0+gitcd9c6e9
# torch cuda version: 12.0
# torch git version: cd9c6e9408dd79175712223895eed36dbdc84f84


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Fri_Jan__6_16:45:21_PST_2023
# Cuda compilation tools, release 12.0, V12.0.140
# Build cuda_12.0.r12.0/compiler.32267302_0

# GPU Hardware Info:
# NVIDIA PG509-210 : 8


exported_program = torch.export.load('/data/users/shangdiy/pytorch/torch_compile_debug/run_2024_11_25_13_59_33_102283-pid_3658904/minifier/checkpoints/exported_program.pt2')
# print(exported_program.graph)
config_patches={'aot_inductor.package': True}
if __name__ == '__main__':
    from torch._dynamo.repro.aoti import run_repro
    with torch.no_grad():
        run_repro(exported_program, config_patches=config_patches, accuracy=False, command='run', save_dir='/data/users/shangdiy/pytorch/torch_compile_debug/run_2024_11_25_13_59_33_102283-pid_3658904/minifier/checkpoints', check_str=None)
```