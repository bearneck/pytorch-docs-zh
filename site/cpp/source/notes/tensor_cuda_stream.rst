Tensor CUDA Stream API
======================

`CUDA Stream`_ 是属于特定 CUDA 设备的线性执行序列。
PyTorch C++ API 通过 CUDAStream 类和有用的辅助函数支持 CUDA 流，使得流操作变得简单。
你可以在 `CUDAStream.h`_ 中找到它们。本文档提供了关于如何使用 PyTorch C++ CUDA Stream API 的更多细节。

.. _CUDA Stream: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams
.. _CUDAStream.h: https://pytorch.org/cppdocs/api/file_c10_cuda_CUDAStream.h.html#file-c10-cuda-cudastream-h
.. _CUDAStreamGuard.h: https://pytorch.org/cppdocs/api/structc10_1_1cuda_1_1_c_u_d_a_stream_guard.html

获取 CUDA 流
*********************

PyTorch 的 C++ API 提供了以下方式来获取 CUDA 流：

1. 从 CUDA 流池中获取一个新流，流是从池中预先分配的，并以轮询方式返回。

.. code-block:: cpp

  CUDAStream getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

.. tip::

  你可以通过将 `isHighPriority` 设置为 `true` 来从高优先级池中请求一个流，或者通过设置设备索引（默认为当前 CUDA 流的设备索引）来为特定设备请求一个流。

2. 获取指定 CUDA 设备的默认 CUDA 流，如果未传递设备索引，则获取当前设备的默认流。

.. code-block:: cpp

  CUDAStream getDefaultCUDAStream(DeviceIndex device_index = -1);

.. tip::

  默认流是当你没有显式使用流时，大多数计算发生的地方。

3. 获取索引为 ``device_index`` 的 CUDA 设备的当前 CUDA 流，如果未传递设备索引，则获取当前设备的当前流。

.. code-block:: cpp

  CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1);

.. tip::

  当前 CUDA 流通常是该设备的默认 CUDA 流，但如果有人调用了 ``setCurrentCUDAStream`` 或使用了 ``StreamGuard`` 或 ``CUDAStreamGuard``，则可能不同。



设置 CUDA 流
***************

PyTorch 的 C++ API 提供了以下方式来设置 CUDA 流：

1. 将传入流所属设备上的当前流设置为传入的流。

.. code-block:: cpp

  void setCurrentCUDAStream(CUDAStream stream);

.. attention::

  此函数可能与当前设备无关。它只改变流所属设备上的当前流。
  我们建议使用 ``CUDAStreamGuard``，因为它会切换到流的设备并使其成为该设备上的当前流。
  ``CUDAStreamGuard`` 还会在销毁时恢复当前设备和流。

2. 使用 ``CUDAStreamGuard`` 在作用域内切换到某个 CUDA 流，它定义在 `CUDAStreamGuard.h`_ 中。

.. tip::

  如果你需要在多个 CUDA 设备上设置流，请使用 ``CUDAMultiStreamGuard``。

CUDA 流使用示例
**************************

1. 在同一设备上获取和设置 CUDA 流

.. code-block:: cpp

  // 此示例展示了如何在同一设备上获取和设置 CUDA 流。
  // 使用 `at::cuda::setCurrentCUDAStream` 来设置当前 CUDA 流

  // 在设备 0 上创建一个张量
  torch::Tensor tensor0 = torch::ones({2, 2}, torch::device(torch::kCUDA));
  // 从设备 0 的 CUDA 流池中获取一个新 CUDA 流
  at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
  // 在设备 0 上将当前 CUDA 流从默认流设置为 `myStream`
  at::cuda::setCurrentCUDAStream(myStream);
  // tensor0 上的 sum() 操作使用 `myStream` 作为当前 CUDA 流
  tensor0.sum();

  // 获取设备 0 上的默认 CUDA 流
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  // 在设备 0 上将当前 CUDA 流设置回默认 CUDA 流
  at::cuda::setCurrentCUDAStream(defaultStream);
  // tensor0 上的 sum() 操作使用 `defaultStream` 作为当前 CUDA 流
  tensor0.sum();

.. code-block:: cpp

  // 此示例与上一个示例相同，但显式指定了设备
  // 索引并使用 CUDA 流保护器来设置当前 CUDA 流

  // 在设备 0 上创建一个张量
  torch::Tensor tensor0 = torch::ones({2, 2}, torch::device(torch::kCUDA));
  // 从设备 0 的 CUDA 流池中获取一个新流
  at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, 0);
  // 使用 CUDA 流保护器在作用域内将当前 CUDA 流设置为 `myStream`
  {
    at::cuda::CUDAStreamGuard guard(myStream);
    // 从此刻直到括号结束，当前 CUDA 流是 `myStream`。
    // tensor0 上的 sum() 操作使用 `myStream` 作为当前 CUDA 流
    tensor0.sum();
  }
  // CUDA 流保护器销毁后，当前 CUDA 流被重置为默认 CUDA 流
  // tensor0 上的 sum() 操作使用设备 0 上的默认 CUDA 流作为当前 CUDA 流
  tensor0.sum();

.. attention::

  上述代码运行在同一个 CUDA 设备上。`setCurrentCUDAStream` 总是在当前设备上设置当前 CUDA 流，
  但请注意 `setCurrentCUDAStream` 实际上是在传入的 CUDA 流所属的设备上设置当前流。


2. 在多个设备上获取和设置 CUDA 流。

.. code-block:: cpp

  // 此示例展示了如何在两个设备上获取和设置 CUDA 流。

  // 从设备 0 和设备 1 的 CUDA 流池中获取新的 CUDA 流
  at::cuda::CUDAStream myStream0 = at::cuda::getStreamFromPool(false, 0);
  at::cuda::CUDAStream myStream1 = at::cuda::getStreamFromPool(false, 1);

  // 在设备 0 上将当前 CUDA 流设置为 `myStream0`
  at::cuda::setCurrentCUDAStream(myStream0);
  // 在设备 1 上将当前 CUDA 流设置为 `myStream1`
  at::cuda::setCurrentCUDAStream(myStream1);

  // 在设备 0 上创建一个张量，无需指定设备索引，因为
  // 当前设备索引是 0
  torch::Tensor tensor0 = torch::ones({2, 2}, torch::device(at::kCUDA));
  // tensor0 上的 sum() 操作在设备 0 上使用 `myStream0` 作为当前 CUDA 流
  tensor0.sum();

// 通过在大括号作用域内使用 CUDA 设备保护器，将当前设备索引更改为 1
{
  at::cuda::CUDAGuard device_guard{1};
  // 在设备 1 上创建一个张量
  torch::Tensor tensor1 = torch::ones({2, 2}, torch::device(at::kCUDA));
  // 在张量 1 上执行 sum() 操作时，使用 `myStream1` 作为设备 1 上的当前 CUDA 流
  tensor1.sum();
}

// 当 device_guard 被销毁后，当前设备重置为设备 0

// 在设备 1 上获取一个新的 CUDA 流
at::cuda::CUDAStream myStream1_1 = at::cuda::getStreamFromPool(false, 1);
// 在设备 1 上创建一个新的张量
torch::Tensor tensor1 = torch::ones({2, 2}, torch::device({torch::kCUDA, 1}));

// 在作用域内使用 CUDA 流保护器，将当前设备索引更改为 1，并将设备 1 上的当前 CUDA 流更改为 `myStream1_1`
{
  at::cuda::CUDAStreamGuard stream_guard(myStream1_1);
  // 在张量 tensor1 上执行 sum() 操作时，使用 `myStream1_1` 作为设备 1 上的当前 CUDA 流
  tensor1.sum();
}

// 当前设备重置为设备 0，且设备 1 上的当前 CUDA 流重置为 `myStream1`

// 在张量 tensor1 上执行 sum() 操作时，使用 `myStream1` 作为设备 1 上的当前 CUDA 流
tensor1.sum();


3. 使用 CUDA 多流保护器

.. code-block:: cpp

  // 此示例展示了如何使用 CUDA 多流保护器同时设置两个设备上的两个流。

  // 创建两个张量，一个在设备 0 上，一个在设备 1 上
  torch::Tensor tensor0 = torch::ones({2, 2}, torch::device({torch::kCUDA, 0}));
  torch::Tensor tensor1 = torch::ones({2, 2}, torch::device({torch::kCUDA, 1}));

  // 从设备 0 和设备 1 的 CUDA 流池中获取新的 CUDA 流
  at::cuda::CUDAStream myStream0 = at::cuda::getStreamFromPool(false, 0);
  at::cuda::CUDAStream myStream1 = at::cuda::getStreamFromPool(false, 1);

  // 使用多流保护器，将设备 0 上的当前 CUDA 流设置为 `myStream0`，
  // 并将设备 1 上的当前 CUDA 流设置为 `myStream1`
  {
    at::cuda::CUDAMultiStreamGuard multi_guard({myStream0, myStream1});

    // 在张量 tensor0 上执行 sum() 操作时，使用 `myStream0` 作为设备 0 上的当前 CUDA 流
    tensor0.sum();
    // 在张量 tensor1 上执行 sum() 操作时，使用 `myStream1` 作为设备 1 上的当前 CUDA 流
    tensor1.sum();
  }

  // 设备 0 上的当前 CUDA 流重置为设备 0 上的默认 CUDA 流
  // 设备 1 上的当前 CUDA 流重置为设备 1 上的默认 CUDA 流

  // 在张量 tensor0 上执行 sum() 操作时，使用默认 CUDA 流作为设备 0 上的当前 CUDA 流
  tensor0.sum();
  // 在张量 tensor1 上执行 sum() 操作时，使用默认 CUDA 流作为设备 1 上的当前 CUDA 流
  tensor1.sum();

.. attention::
  ``CUDAMultiStreamGuard`` 不会改变当前设备索引，它只改变每个传入流所在设备上的流。
  除了作用域控制外，此保护器等同于在每个传入流上调用 ``setCurrentCUDAStream``。

4. 在多设备上处理 CUDA 流的框架示例

.. code-block:: cpp

   // 这是一个框架示例，展示了如何在多个设备上处理 CUDA 流
   // 假设你想同时在两个设备的非默认流上工作，并且我们
   // 已经有两个向量分别包含两个设备上的流。以下代码展示了三种
   // 获取和设置流的方式。

   // 用法 0：使用 `setCurrentCUDAStream` 获取 CUDA 流并设置当前 CUDA 流
   // 在设备 0 上创建一个 CUDA 流向量 `streams0`
   std::vector<at::cuda::CUDAStream> streams0 =
     {at::cuda::getDefaultCUDAStream(), at::cuda::getStreamFromPool()};
   // 将设备 0 上的当前流设置为 `streams0[0]`
   at::cuda::setCurrentCUDAStream(streams0[0]);

   // 使用 CUDA 设备保护器在设备上创建一个 CUDA 流向量 `streams1`
   std::vector<at::cuda::CUDAStream> streams1;
   {
     // 在此作用域内，设备索引设置为 1
     at::cuda::CUDAGuard device_guard(1);
     streams1.push_back(at::cuda::getDefaultCUDAStream());
     streams1.push_back(at::cuda::getStreamFromPool());
   }
   // 当 device_guard 被销毁后，设备索引重置为 0

   // 将设备 1 上的当前流设置为 `streams1[0]`
   at::cuda::setCurrentCUDAStream(streams1[0]);


   // 用法 1：使用 CUDA 设备保护器仅更改当前设备索引
   {
     at::cuda::CUDAGuard device_guard(1);

     // 在作用域内，当前设备索引更改为 1
     // 设备 1 上的当前 CUDA 流仍然是 `streams1[0]`，没有变化
   }
   // 当 `device_guard` 被销毁后，当前设备索引重置为 0


   // 用法 2：使用 CUDA 流保护器同时更改当前设备索引和当前 CUDA 流。
   {
     at::cuda::CUDAStreamGuard stream_guard(streams1[1]);

     // 在作用域内，当前设备索引和当前 CUDA 流被设置为 1 和 `streams1[1]`
   }
   // 当 stream_guard 被销毁后，当前设备索引和当前 CUDA 流重置为 0 和 `streams0[0]`


   // 用法 3：使用 CUDA 多流保护器更改多个设备上的多个流
   {
     // 这等同于在两个流上都调用 `torch::cuda::setCurrentCUDAStream`
     at::cuda::CUDAMultiStreamGuard multi_guard({streams0[1], streams1[1]});

     // 当前设备索引没有改变，仍然是 0
     // 设备 0 和设备 1 上的当前 CUDA 流被设置为 `streams0[1]` 和 `streams1[1]`
   }
   // 当 `multi_guard` 被销毁后，设备 0 和设备 1 上的当前 CUDA 流重置为 `streams0[0]` 和 `streams1[0]`。