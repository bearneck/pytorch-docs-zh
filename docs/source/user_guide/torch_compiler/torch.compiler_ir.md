
# IRs

PyTorch 2.0 为后端提供了两组 IR 进行交互：Core Aten IR 和 Prims IR。

## Core Aten IR

Core aten 操作符是 aten 操作符的核心子集，可用于组合其他操作符。
Core aten IR 是完全函数式的，此操作集中没有 `inplace` 或 `_out` 变体。
与 Prims IR 相比，core aten 操作复用了 "native_functions.yaml" 中现有的 aten 操作，
并且它不会将操作进一步分解为显式的类型提升和广播操作。
此操作集旨在作为与后端交互的函数式 IR。

```{warning}
  此操作集仍在积极开发中，未来将添加更多操作。
```

```{csv-table}
   :file: ../../../build/ir/aten_ops.csv
   :widths: auto
   :header-rows: 1
```

## Prims IR

Prims IR 是一组可用于组合其他操作符的原始操作符。
Prims IR 是比 core aten IR 更低级的操作集，它将操作进一步分解为显式的
类型提升和广播操作：prims.convert_element_type 和 prims.broadcast_in_dim。
此操作集旨在与编译器后端交互。

```{warning}
  此操作集仍在积极开发中，未来将添加更多操作。
```

```{csv-table}
   :file: ../../../build/ir/prims_ops.csv
   :widths: auto
   :header-rows: 1
```