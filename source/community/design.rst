:orphan:

PyTorch 设计理念
=========================

本文档旨在帮助贡献者和模块维护者理解 PyTorch 在发展过程中形成的高层设计原则。这些原则并非硬性规定，而是作为指导方针，帮助权衡不同考量点，并解决在开发 PyTorch 过程中可能出现的分歧。有关贡献、模块维护以及如何将分歧升级至核心维护者的更多信息，请参阅 `PyTorch 治理 <https://pytorch.org/docs/main/community/governance.html>`__。

设计原则
-----------------

原则一：可用性优先于性能
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这条原则可能令人惊讶！正如一位 Hacker News 发帖者写道：*PyTorch 太棒了！[...] 但我很困惑。一个机器学习框架怎么能不痴迷于速度/性能呢？* 参见 `Hacker News 上关于 PyTorch 的讨论 <https://news.ycombinator.com/item?id=28066093>`__。

Soumith 关于 `发展 PyTorch 社区 <https://soumith.ch/posts/2021/02/growing-opensource/?fbclid=IwAR1bvN_xZ8avGvu14ODJzS8Zp7jX1BOyfuGUf-zoRawpyL-s95Vjxf88W7s>`__ 的博客文章对此进行了深入探讨，但概括来说：

-  PyTorch 的首要目标是可用性
-  次要目标是拥有*合理的*性能

我们相信，保持灵活性以支持在我们抽象之上进行研究工作的能力仍然至关重要。我们无法预知未来的工作负载会是什么样子，但我们知道我们希望它们首先在 PyTorch 上构建，而这需要灵活性。

更具体地说，我们以*可用性优先*的方式运作，并尽量避免在没有清晰权衡视角的情况下，贸然转向*限制优先*的模式（例如，静态形状、仅图模式）。通常，人们倾向于预先施加严格的用户限制，因为这可以简化实现，但这伴随着风险：

-  性能的提升可能不值得用户为此付出的代价，要么是因为性能优势不够显著，要么是它只适用于相对狭窄的子问题集。
-  即使性能优势显著，这些限制也可能将生态系统分割成具有不同限制条件的集合，这些限制条件可能很快让用户难以理解。

我们希望用户能够无缝地将他们的 PyTorch 代码迁移到不同的硬件和软件平台，与不同的库和框架互操作，并体验 PyTorch 用户体验的全部丰富性，而不是一个最小公分母的子集。

原则二：简单优于便捷
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这里，我们借鉴了 `Python 之禅 <https://peps.python.org/pep-0020/>`__：

-  *显式优于隐式*
-  *简单优于复杂*

描述这两个目标的一个更简洁的方式是 `简单优于便捷 <https://www.infoq.com/presentations/Simple-Made-Easy/>`_。让我们从一个例子开始，因为在日常英语中*简单*和*便捷*经常互换使用。考虑如何在 PyTorch 中建模 `设备 <https://pytorch.org/docs/main/tensor_attributes.html#torch.device>`__：

-  **简单 / 显式（易于理解、调试）：** 每个张量都与一个设备相关联。用户显式指定张量的设备移动。需要跨设备移动的操作会导致错误。
-  **便捷 / 隐式（易于使用）：** 用户无需担心设备；系统会找出全局最优的设备放置方案。

在这个具体案例中，并且作为一般的设计理念，PyTorch 倾向于暴露简单且显式的构建块，而不是对从业者来说易于使用的 API。简单的版本对于新的 PyTorch 用户来说是立即可以理解和调试的：如果你在程序中实际调用需要跨设备移动的运算符时，你会得到一个清晰的错误。便捷的解决方案可能让新用户最初上手更快，但调试这样的系统可能很复杂：系统是如何做出决定的？接入这样的系统的 API 是什么？对象在其 IR 中是如何表示的？

支持这种设计的一些经典论点来自 `关于分布式计算的说明 <https://dl.acm.org/doi/book/10.5555/974938>`__（TLDR：不要对性能特征差异很大的资源进行统一建模，细节总会暴露出来）和 `端到端原则 <http://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf>`__（TLDR：在堆栈的底层构建智能可能会阻碍在堆栈高层构建高性能特性，而且通常无论如何也行不通）。例如，我们可以构建运算符级别或全局的设备移动规则，但精确的选择并不明显，并且构建一个可扩展的机制具有不可避免的复杂性和延迟成本。

这里需要注意的是，这并不意味着高层的“便捷”API 没有价值；当然，例如，在堆栈的高层支持跨大型集群异构计算的高效张量计算是有价值的。相反，我们的意思是，专注于简单的底层构建块有助于为便捷的 API 提供信息，同时当用户需要偏离常规路径时，仍然保持良好的体验。这也为创新和更具倾向性的工具以我们无法在 PyTorch 核心库中支持的速度增长留下了空间，但最终会从中受益，正如我们 `丰富的生态系统 <https://pytorch.org/ecosystem/>`__ 所证明的那样。换句话说，一开始不自动化，可能让我们更快地达到良好自动化的水平。

原则三：Python 优先，并提供一流的语言互操作性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这条原则始于 **Python 优先**：

PyTorch 并非一个将单一 C++ 框架封装为 Python 绑定的库。
它的构建目标是深度集成到 Python 生态中。你可以像使用 `NumPy <https://www.numpy.org/>`__、
`SciPy <https://www.scipy.org/>`__、`scikit-learn <https://scikit-learn.org/>`__
或其他 Python 库一样自然地使用它。你可以直接用 Python 编写新的神经网络层，
利用你喜爱的库以及诸如 `Cython <https://cython.org/>`__ 和
`Numba <http://numba.pydata.org/>`__ 这样的工具包。我们的目标是在适当之处避免重复造轮子。

多年来 PyTorch 需要应对的一个问题是 Python 开销：我们首先用 C++ 重写了 `autograd` 引擎，
然后是大部分算子定义，接着开发了 TorchScript 和 C++ 前端。

尽管如此，在 Python 环境中工作依然能为用户提供最佳体验：它灵活、熟悉，并且或许最重要的是，
拥有庞大的科学计算库和扩展生态系统可供使用。这一事实推动了我们最近的一些贡献，
旨在达到接近 Python 可用性曲线末端的帕累托最优点：

-  `TorchDynamo <https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361>`__，
   一个能够以最小用户干预加速现有即时执行模式 PyTorch 程序的 Python 帧评估工具。
-  `torch_function <https://pytorch.org/docs/main/notes/extending.html#extending-torch>`__
   和 `torch_dispatch <https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557>`__
   扩展点，使得基于 C++ 内部实现构建以 Python 为先的功能成为可能，例如分别实现的
   `torch.fx 追踪器 <https://pytorch.org/docs/stable/fx.html>`__
   和 `functorch <https://github.com/pytorch/functorch>`__。

这些设计原则并非僵化的教条，而是经过实践检验的选择，它们锚定了我们构建 PyTorch 的方式，
使其成为如今这个可调试、可定制且灵活的框架。随着贡献者和维护者队伍的壮大，
我们期待与您一起将这些核心原则应用到我们的库和生态系统中。我们也愿意随着认知的深入和 AI 领域的发展而不断演进这些原则，
因为我们深知这一领域必将持续变化。