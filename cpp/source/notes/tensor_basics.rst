Tensor Basics
=============

支撑 PyTorch 的 ATen 张量库是一个简单的张量库，它直接在 C++17 中公开了 Torch 中的张量操作。ATen 的 API 是从 PyTorch 使用的相同声明自动生成的，因此这两个 API 会随时间推移保持同步。

张量类型是动态解析的，因此 API 是通用的，不包含模板。也就是说，只有一个 ``Tensor`` 类型。它可以容纳 CPU 或 CUDA 张量，并且张量可以包含 Double、Float、Int 等类型。这种设计使得编写通用代码变得容易，而无需对所有内容进行模板化。

有关提供的 API，请参阅 https://pytorch.org/cppdocs/api/namespace_at.html#functions。摘录如下：

.. code-block:: cpp

  Tensor atan2(const Tensor & other) const;
  Tensor & atan2_(const Tensor & other);
  Tensor pow(Scalar exponent) const;
  Tensor pow(const Tensor & exponent) const;
  Tensor & pow_(Scalar exponent);
  Tensor & pow_(const Tensor & exponent);
  Tensor lerp(const Tensor & end, Scalar weight) const;
  Tensor & lerp_(const Tensor & end, Scalar weight);
  Tensor histc() const;
  Tensor histc(int64_t bins) const;
  Tensor histc(int64_t bins, Scalar min) const;
  Tensor histc(int64_t bins, Scalar min, Scalar max) const;

还提供了原地操作，并且总是以 `_` 作为后缀，表示它们将修改张量。

高效访问张量元素
-----------------------------------

当使用张量范围的操作时，动态分派的相对成本非常小。然而，在某些情况下，特别是在您自己的内核中，需要高效的元素级访问，而元素级循环内部的动态分派成本非常高。ATen 提供了 *访问器*，它们通过一次动态检查创建，以确认张量的类型和维度数量。然后，访问器公开一个 API，用于高效地访问张量元素。

访问器是张量的临时视图。它们仅在它们所查看的张量的生命周期内有效，因此应仅在函数的局部使用，就像迭代器一样。

请注意，访问器与内核函数内部的 CUDA 张量不兼容。相反，您必须使用 *打包访问器*，它的行为方式相同，但复制张量元数据而不是指向它。

因此，建议对 CPU 张量使用 *访问器*，对 CUDA 张量使用 *打包访问器*。

CPU 访问器
*************

.. code-block:: cpp

  torch::Tensor foo = torch::rand({12, 12});

  // 断言 foo 是二维的并包含浮点数。
  auto foo_a = foo.accessor<float,2>();
  float trace = 0;

  for(int i = 0; i < foo_a.size(0); i++) {
    // 使用访问器 foo_a 获取张量数据。
    trace += foo_a[i][i];
  }

CUDA 访问器
**************


.. code-block:: cpp

  __global__ void packed_accessor_kernel(
      torch::PackedTensorAccessor64<float, 2> foo,
      float* trace) {
    int i = threadIdx.x;
    gpuAtomicAdd(trace, foo[i][i]);
  }

  torch::Tensor foo = torch::rand({12, 12});

  // 断言 foo 是二维的并包含浮点数。
  auto foo_a = foo.packed_accessor64<float,2>();
  float trace = 0;

  packed_accessor_kernel<<<1, 12>>>(foo_a, &trace);

除了 ``PackedTensorAccessor64`` 和 ``packed_accessor64`` 之外，还有相应的 ``PackedTensorAccessor32`` 和 ``packed_accessor32``，它们使用 32 位整数进行索引。这在 CUDA 上可能会快很多，但可能导致索引计算溢出。

请注意，模板可以包含其他参数，例如指针限制和用于索引的整数类型。有关 *访问器* 和 *打包访问器* 的完整模板描述，请参阅文档。

使用外部创建的数据
-----------------------------

如果您已经在内存（CPU 或 CUDA）中分配了张量数据，您可以将该内存视为 ATen 中的 ``Tensor``：

.. code-block:: cpp

  float data[] = { 1, 2, 3,
                   4, 5, 6 };
  torch::Tensor f = torch::from_blob(data, {2, 3});

这些张量无法调整大小，因为 ATen 不拥有该内存，但在其他方面行为与普通张量相同。

标量和零维张量
------------------------------------

除了 ``Tensor`` 对象，ATen 还包括表示单个数字的 ``Scalar``。与张量类似，标量是动态类型的，可以容纳 ATen 的任何数字类型之一。标量可以从 C++ 数字类型隐式构造。需要标量是因为某些函数（如 ``addmm``）接受数字和张量，并期望这些数字与张量具有相同的动态类型。它们也用于 API 中，以指示函数将 *始终* 返回标量值的位置，例如 ``sum``。

.. code-block:: cpp

  namespace torch {
  Tensor addmm(Scalar beta, const Tensor & self,
               Scalar alpha, const Tensor & mat1,
               const Tensor & mat2);
  Scalar sum(const Tensor & self);
  } // namespace torch

  // 用法。
  torch::Tensor a = ...
  torch::Tensor b = ...
  torch::Tensor c = ...
  torch::Tensor r = torch::addmm(1.0, a, .5, b, c);

除了 ``Scalar``，ATen 还允许 ``Tensor`` 对象是零维的。这些张量包含单个值，并且它们可以是对更大 ``Tensor`` 中单个元素的引用。它们可以在任何需要 ``Tensor`` 的地方使用。它们通常由像 `select` 这样的操作符创建，这些操作符减少了 ``Tensor`` 的维度。

.. code-block:: cpp

  torch::Tensor two = torch::rand({10, 20});
  two[1][2] = 4;
  // ^^^^^^ <- 零维张量