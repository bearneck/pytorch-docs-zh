 currentmodule
torch.cuda.\_sanitizer


# CUDA 流消毒器

 note
 title
Note


这是一个原型功能，意味着它处于早期反馈和测试阶段，其组件可能会发生变化。


## 概述

 automodule
torch.cuda.\_sanitizer


## 使用方法

以下是一个在 PyTorch 中简单的同步错误示例：

    import torch

    a = torch.rand(4, 2, device="cuda")

    with torch.cuda.stream(torch.cuda.Stream()):
        torch.mul(a, 5, out=a)

张量 `a` 在默认流上初始化，然后在没有任何同步方法的情况下，在一个新流上被修改。这两个内核将在同一个张量上并发运行，这可能导致第二个内核在第一个内核完成写入之前读取到未初始化的数据，或者第一个内核可能覆盖第二个内核的部分结果。 当在命令行中使用以下命令运行此脚本时： :

    TORCH_CUDA_SANITIZER=1 python example_error.py

CSAN 会打印以下输出：

    ============================
    CSAN 在数据指针为 139719969079296 的张量上检测到可能的数据竞争
    在内核执行期间，流 94646435460352 的访问：
    aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    写入参数 self, out 以及输出
    堆栈跟踪：
      File "example_error.py", line 6, in <module>
        torch.mul(a, 5, out=a)
      ...
      File "pytorch/torch/cuda/_sanitizer.py", line 364, in _handle_kernel_launch
        stack_trace = traceback.StackSummary.extract(

    先前在内核执行期间，流 0 的访问：
    aten::rand(int[] size, *, int? dtype=None, Device? device=None) -> Tensor
    写入输出
    堆栈跟踪：
      File "example_error.py", line 3, in <module>
        a = torch.rand(10000, device="cuda")
      ...
      File "pytorch/torch/cuda/_sanitizer.py", line 364, in _handle_kernel_launch
        stack_trace = traceback.StackSummary.extract(

    张量分配的堆栈跟踪：
      File "example_error.py", line 3, in <module>
        a = torch.rand(10000, device="cuda")
      ...
      File "pytorch/torch/cuda/_sanitizer.py", line 420, in _handle_memory_allocation
        traceback.StackSummary.extract(

这提供了关于错误来源的详细信息：

- 一个张量被从 ID 为 0（默认流）和 94646435460352（新流）的流错误地访问。

- 该张量是通过调用 `a = torch.rand(10000, device="cuda")` 分配的。

- 

  错误的访问是由以下操作符引起的：

  :   - 在流 0 上的 `a = torch.rand(10000, device="cuda")`
      - 在流 94646435460352 上的 `torch.mul(a, 5, out=a)`

- 错误信息还显示了被调用操作符的模式，并附有说明，指出操作符的哪些参数对应于受影响的张量。
  - 在示例中，可以看到张量 `a` 对应于被调用操作符 `torch.mul` 的参数 `self`、`out` 以及 `output` 值。

 seealso
支持的 torch 操作符及其模式列表可以在此处查看： `here <torch>`{.interpreted-text role="doc"}。


可以通过强制新流等待默认流来修复此错误：

    with torch.cuda.stream(torch.cuda.Stream()):
        torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
        torch.mul(a, 5, out=a)

当脚本再次运行时，不会报告任何错误。

## API 参考

 autofunction
enable_cuda_sanitizer


 autofunction
zip_arguments


 autofunction
zip_by_key

