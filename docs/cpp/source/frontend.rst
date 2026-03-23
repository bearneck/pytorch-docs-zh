C++ 前端
================

PyTorch C++ 前端是一个用于 CPU 和 GPU 张量计算的 C++17 库，具备自动微分功能，并为前沿机器学习应用提供高级构建模块。

描述
-----------

PyTorch C++ 前端可被视为 PyTorch Python 前端的 C++ 版本，为机器学习和神经网络提供自动微分及各种高级抽象。具体而言，它包含以下组件：

+----------------------+------------------------------------------------------------------------+
| 组件                 | 描述                                                                  |
+======================+========================================================================+
| ``torch::Tensor``    | 支持自动微分、高效的 CPU 和 GPU 张量                                  |
+----------------------+------------------------------------------------------------------------+
| ``torch::nn``        | 用于神经网络建模的可组合模块集合                                       |
+----------------------+------------------------------------------------------------------------+
| ``torch::optim``     | 用于训练模型的优化算法，如 SGD、Adam 或 RMSprop                       |
+----------------------+------------------------------------------------------------------------+
| ``torch::data``      | 数据集、数据管道以及多线程异步数据加载器                              |
+----------------------+------------------------------------------------------------------------+
| ``torch::serialize`` | 用于存储和加载模型检查点的序列化 API                                  |
+----------------------+------------------------------------------------------------------------+
| ``torch::python``    | 将 C++ 模型绑定到 Python 的粘合层                                     |
+----------------------+------------------------------------------------------------------------+
| ``torch::jit``       | 对 TorchScript JIT 编译器的纯 C++ 访问                                |
+----------------------+------------------------------------------------------------------------+

端到端示例
------------------

以下是一个在 MNIST 数据集上定义并训练简单神经网络的完整端到端示例：

.. code-block:: cpp

  #include <torch/torch.h>

  // 定义一个新的模块。
  struct Net : torch::nn::Module {
    Net() {
      // 构造并注册两个 Linear 子模块。
      fc1 = register_module("fc1", torch::nn::Linear(784, 64));
      fc2 = register_module("fc2", torch::nn::Linear(64, 32));
      fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    // 实现 Net 的算法。
    torch::Tensor forward(torch::Tensor x) {
      // 使用众多张量操作函数之一。
      x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
      x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
      x = torch::relu(fc2->forward(x));
      x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
      return x;
    }

    // 使用众多“标准库”模块之一。
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  };

  int main() {
    // 创建一个新的 Net。
    auto net = std::make_shared<Net>();

    // 为 MNIST 数据集创建一个多线程数据加载器。
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/64);

    // 实例化一个 SGD 优化算法来更新 Net 的参数。
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
      size_t batch_index = 0;
      // 迭代数据加载器以从数据集中产生批次。
      for (auto& batch : *data_loader) {
        // 重置梯度。
        optimizer.zero_grad();
        // 在输入数据上执行模型。
        torch::Tensor prediction = net->forward(batch.data);
        // 计算损失值以评估模型的预测。
        torch::Tensor loss = torch::nll_loss(prediction, batch.target);
        // 计算损失相对于模型参数的梯度。
        loss.backward();
        // 根据计算出的梯度更新参数。
        optimizer.step();
        // 每 100 个批次输出损失并保存检查点。
        if (++batch_index % 100 == 0) {
          std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                    << " | Loss: " << loss.item<float>() << std::endl;
          // 定期将模型序列化为检查点。
          torch::save(net, "net.pt");
        }
      }
    }
  }

要查看更多使用 PyTorch C++ 前端的完整示例，请参阅 `示例仓库
<https://github.com/pytorch/examples/tree/master/cpp>`_。

设计理念
----------

PyTorch C++ 前端的设计理念是：Python 前端非常出色，应尽可能使用；但在某些场景下，性能和可移植性要求使得使用 Python 解释器不可行。例如，对于低延迟、高性能或多线程环境（如视频游戏或生产服务器），Python 是一个较差的选择。C++ 前端的目标是解决这些用例，同时不牺牲 Python 前端的用户体验。

因此，C++ 前端在开发时遵循了以下设计理念：

* **在设计、命名、约定和功能上紧密模拟 Python 前端**。虽然两个前端之间可能存在偶尔的差异（例如，我们可能移除了已弃用的功能或修复了 Python 前端中的“瑕疵”），但我们保证将 Python 模型移植到 C++ 的工作应仅在于**翻译语言特性**，而不是修改功能或行为。

* **优先考虑灵活性和用户友好性，而非微观优化。**
  在 C++ 中，你通常可以获得最优的代码，但代价是极其不友好的用户体验。灵活性和动态性是 PyTorch 的核心，C++ 前端致力于保留这种体验，有时会牺牲性能（或“隐藏”性能调节选项）以保持 API 的简洁和易于理解。我们希望那些不以编写 C++ 为生的研究人员也能够使用我们的 API。

请注意：Python 不一定比 C++ 慢！Python 前端几乎将所有计算密集型任务（尤其是任何类型的数值运算）都调用到 C++ 中，而这些操作将占据程序运行时间的大部分。如果你更倾向于编写 Python，并且能够承担编写 Python 的成本，我们建议使用 PyTorch 的 Python 接口。然而，如果你更倾向于编写 C++，或者需要编写 C++（由于多线程、延迟或部署要求），PyTorch 的 C++ 前端提供了一个与其 Python 对应部分大致同样方便、灵活、友好和直观的 API。这两个前端服务于不同的用例，协同工作，并非旨在无条件地相互替代。

安装
------------

关于如何安装 C++ 前端库发行版的说明，包括如何构建一个依赖于 LibTorch 的最小应用程序的示例，可以通过点击 `此链接 <https://pytorch.org/cppdocs/installing.html>`_ 找到。