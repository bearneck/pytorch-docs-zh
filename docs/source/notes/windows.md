# Windows 常见问题解答

## 从源码构建

### 包含可选组件

Windows 版 PyTorch 支持两个组件： MKL 和 MAGMA。以下是使用它们进行构建的步骤。

``` bat
REM 确保已安装 7z 和 curl。

REM 下载 MKL 文件
curl https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z -k -O
7z x -aoa mkl_2020.2.254.7z -omkl

REM 下载 MAGMA 文件
REM 可用版本：
REM 2.5.4 (CUDA 10.1 10.2 11.0 11.1) x (Debug Release)
REM 2.5.3 (CUDA 10.1 10.2 11.0) x (Debug Release)
REM 2.5.2 (CUDA 9.2 10.0 10.1 10.2) x (Debug Release)
REM 2.5.1 (CUDA 9.2 10.0 10.1 10.2) x (Debug Release)
set "CUDA_PREFIX=cuda102"
set "CONFIG=release"
set "HOST=https://s3.amazonaws.com/ossci-windows"
curl -k "%HOST%/magma_2.5.4_%CUDA_PREFIX%_%CONFIG%.7z" -o magma.7z
7z x -aoa magma.7z -omagma

REM 设置必要的环境变量
set "CMAKE_INCLUDE_PATH=%cd%\mkl\include"
set "LIB=%cd%\mkl\lib;%LIB%"
set "MAGMA_HOME=%cd%\magma"
```

### 加速 Windows 上的 CUDA 构建

Visual Studio 目前不支持并行自定义任务。 作为替代方案，我们可以使用 `Ninja` 来并行化 CUDA 构建任务。只需输入几行代码即可使用它。

``` bat
REM 首先安装 ninja。
pip install ninja

REM 将其设置为 cmake 生成器
set CMAKE_GENERATOR=Ninja
```

### 一键安装脚本

你可以查看 [这套脚本](https://github.com/peterjc123/pytorch-scripts)。 它将为你指明方向。

## 扩展

### CFFI 扩展

对 CFFI 扩展的支持非常实验性。你必须在 `Extension` 对象中指定额外的 `libraries` 才能使其在 Windows 上构建。

``` python
ffi = create_extension(
    '_ext.my_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_compile_args=["-std=c99"],
    libraries=['ATen', '_C'] # 必要时附加 cuda 库，例如 cudart
)
```

### Cpp 扩展

与上一种类型相比，这种扩展有更好的支持。 然而，它仍然需要一些手动配置。首先，你应该打开 **适用于 VS 2017 的 x86_x64 交叉工具命令提示符**。 然后，你就可以开始编译过程了。

## 安装

### 在 win-32 频道中找不到包。

``` bat
Solving environment: failed

PackagesNotFoundError: The following packages are not available from current channels:

- pytorch

Current channels:
- https://repo.continuum.io/pkgs/main/win-32
- https://repo.continuum.io/pkgs/main/noarch
- https://repo.continuum.io/pkgs/free/win-32
- https://repo.continuum.io/pkgs/free/noarch
- https://repo.continuum.io/pkgs/r/win-32
- https://repo.continuum.io/pkgs/r/noarch
- https://repo.continuum.io/pkgs/pro/win-32
- https://repo.continuum.io/pkgs/pro/noarch
- https://repo.continuum.io/pkgs/msys2/win-32
- https://repo.continuum.io/pkgs/msys2/noarch
```

PyTorch 不能在 32 位系统上运行。请使用 Windows 和 Python 的 64 位版本。

### 导入错误

``` python
from torch._C import *

ImportError: DLL load failed: The specified module could not be found.
```

该问题是由缺少必要文件引起的。 对于 wheel 包，由于我们没有打包一些库和 VS2017 可再发行文件，请确保你手动安装它们。 可以下载 [VS 2017 可再发行安装程序](https://aka.ms/vs/15/release/VC_redist.x64.exe)。 你还应该注意你的 Numpy 安装。确保它 使用 MKL 而不是 OpenBLAS。你可以输入以下命令。

``` bat
pip install numpy mkl intel-openmp mkl_fft
```

## 使用（多进程）

### 没有 if 子句保护导致的多进程错误

``` python
RuntimeError:
       An attempt has been made to start a new process before the
       current process has finished its bootstrapping phase.

   This probably means that you are not using fork to start your
   child processes and you have forgotten to使用 the proper idiom
   in the main module:

       if __name__ == '__main__':
           freeze_support()
           ...

   The "freeze_support()" line can be omitted if the program
   is not going to be frozen to produce an executable.
```

Windows 上 `multiprocessing` 的实现方式不同，它 使用 `spawn` 而不是 `fork`。因此，我们必须用 if 子句包装代码，以防止代码多次执行。将 你的代码重构为以下结构。

``` python
import torch

def main()
    for i, data in enumerate(dataloader):
        # 在这里做一些事情

if __name__ == '__main__':
    main()
```

### 多进程错误 \"Broken pipe\"

``` python
ForkingPickler(file, protocol).dump(obj)

BrokenPipeError: [Errno 32] Broken pipe
```

当子进程在父进程完成发送数据之前结束时，会发生此问题。 你的代码可能有问题。你可以 通过将 `torch.utils.data.DataLoader` 的 `num_worker` 减少到零来调试你的代码，看看问题是否仍然存在。

### 多进程错误 \"driver shut down\"

    Couldn’t open shared file mapping: <torch_14808_1591070686>, error code: <1455> at torch\lib\TH\THAllocator.c:154

    [windows] driver shut down

请更新您的显卡驱动程序。如果问题持续存在，可能是您的显卡过于老旧，或者计算任务对您的显卡来说过重。请根据这篇 [文章](https://www.pugetsystems.com/labs/hpc/Working-around-TDR-in-Windows-for-a-better-GPU-computing-experience-777/) 更新 TDR 设置。

### CUDA IPC 操作

``` python
THCudaCheck FAIL file=torch\csrc\generic\StorageSharing.cpp line=252 error=63 : OS call failed or operation not supported on this OS
```

Windows 系统不支持这些操作。例如，在 CUDA 张量上进行多进程处理是无法成功的，对此有两种替代方案。

1.  不要使用 `multiprocessing`。将 `torch.utils.data.DataLoader` 的 `num_worker` 设置为零。
2.  改为共享 CPU 张量。确保您的自定义 `torch.utils.data.DataSet` 返回 CPU 张量。
