orphan

:   

# PyTorch 治理 \| 构建 + CI

## 如何新增维护者

要成为维护者，候选人需要满足以下条件：

- 在 PyTorch 仓库的相关部分至少提交六次提交
- 其中至少有一次提交是在过去六个月内完成的

要将符合条件的候选人添加到维护者列表中，请创建一个 PR，将候选人添加到 [关注人员](https://pytorch.org/docs/main/community/persons_of_interest.html) 页面和 [merge_rules](https://github.com/pytorch/pytorch/blob/main/.github/merge_rules.yaml) 文件中。当前的维护者将进行支持投票。批准该 PR 的决策标准如下：

- 合并前至少经过两个工作日（确保大多数贡献者已看到）
- PR 具有正确的标签 ([module: ci]{.title-ref})
- 当前维护者没有提出反对意见
- 获得当前维护者至少三个净 *点赞*（或者当该模块的维护者少于 3 人时，所有维护者都投票 *点赞*）。
