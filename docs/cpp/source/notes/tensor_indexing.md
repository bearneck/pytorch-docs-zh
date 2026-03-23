# Tensor Indexing API

在 PyTorch C++ API 中索引张量的操作与 Python API 非常相似。 所有索引类型，例如 `None` / `...` / 整数 / 布尔值 / 切片 / 张量， 在 C++ API 中均可用，这使得将 Python 索引代码转换为 C++ 非常简单。 主要区别在于，C++ API 中不使用类似于 Python API 语法的 `[]` 运算符， 而是使用以下索引方法：

- `torch::Tensor::index` ([链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor5indexE8ArrayRefIN2at8indexing11TensorIndexEE))
- `torch::Tensor::index_put_` ([链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4N2at6Tensor10index_put_E8ArrayRefIN2at8indexing11TensorIndexEERK6Tensor))

同样需要注意的是，诸如 `None` / `Ellipsis` / `Slice` 等索引类型 位于 `torch::indexing` 命名空间中，建议在任何索引代码之前放置 `using namespace torch::indexing` 以方便使用这些索引类型。

以下是一些将 Python 索引代码转换为 C++ 的示例：

## Getter

  ---------------------------------------------------------------------------------------------------------------------------------
  Python                                               C++ (假设 `using namespace torch::indexing`)
  ---------------------------------------------------- ----------------------------------------------------------------------------
  `tensor[None]`                                       `tensor.index({None})`

  `tensor[Ellipsis, ...]`                              `tensor.index({Ellipsis, "..."})`

  `tensor[1, 2]`                                       `tensor.index({1, 2})`

  `tensor[True, False]`                                `tensor.index({true, false})`

  `tensor[1::2]`                                       `tensor.index({Slice(1, None, 2)})`

  `tensor[torch.tensor([1, 2])]`                       `tensor.index({torch::tensor({1, 2})})`

  `tensor[..., 0, True, 1::2, torch.tensor([1, 2])]`   `tensor.index({"...", 0, true, Slice(1, None, 2), torch::tensor({1, 2})})`
  ---------------------------------------------------------------------------------------------------------------------------------

## Setter

  ---------------------------------------------------------------------------------------------------------------------------------------------
  Python                                                   C++ (假设 `using namespace torch::indexing`)
  -------------------------------------------------------- ------------------------------------------------------------------------------------
  `tensor[None] = 1`                                       `tensor.index_put_({None}, 1)`

  `tensor[Ellipsis, ...] = 1`                              `tensor.index_put_({Ellipsis, "..."}, 1)`

  `tensor[1, 2] = 1`                                       `tensor.index_put_({1, 2}, 1)`

  `tensor[True, False] = 1`                                `tensor.index_put_({true, false}, 1)`

  `tensor[1::2] = 1`                                       `tensor.index_put_({Slice(1, None, 2)}, 1)`

  `tensor[torch.tensor([1, 2])] = 1`                       `tensor.index_put_({torch::tensor({1, 2})}, 1)`

  `tensor[..., 0, True, 1::2, torch.tensor([1, 2])] = 1`   `tensor.index_put_({"...", 0, true, Slice(1, None, 2), torch::tensor({1, 2})}, 1)`
  ---------------------------------------------------------------------------------------------------------------------------------------------

## Python/C++ 索引类型之间的转换

Python 和 C++ 索引类型之间的一一对应关系如下：

  ----------------------------------------------------------------------------------------
  Python                   C++ (假设使用 `using namespace torch::indexing`)
  ------------------------ ---------------------------------------------------------------
  `None`                   `None`

  `Ellipsis`               `Ellipsis`

  `...`                    `"..."`

  `123`                    `123`

  `True`                   `true`

  `False`                  `false`

  `:` 或 `::`              `Slice()` 或 `Slice(None, None)` 或 `Slice(None, None, None)`

  `1:` 或 `1::`            `Slice(1, None)` 或 `Slice(1, None, None)`

  `:3` 或 `:3:`            `Slice(None, 3)` 或 `Slice(None, 3, None)`

  `::2`                    `Slice(None, None, 2)`

  `1:3`                    `Slice(1, 3)`

  `1::2`                   `Slice(1, None, 2)`

  `:3:2`                   `Slice(None, 3, 2)`

  `1:3:2`                  `Slice(1, 3, 2)`

  `torch.tensor([1, 2])`   `torch::tensor({1, 2})`
  ----------------------------------------------------------------------------------------
