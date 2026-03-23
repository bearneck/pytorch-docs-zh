


# 类型信息

一个 `torch.dtype` 的数值属性可以通过 `torch.finfo` 或 `torch.iinfo` 来访问。


## torch.finfo


`torch.finfo` 是一个表示浮点型 `torch.dtype`（例如 ``torch.float32``、``torch.float64``、``torch.float16`` 和 ``torch.bfloat16``）数值属性的对象。这类似于 [numpy.finfo](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)。

一个 `torch.finfo` 提供以下属性：

| 名称              | 类型   | 描述                                                                                 |
| :---------------- | :----- | :----------------------------------------------------------------------------------- |
| bits              | int    | 该类型占用的位数。                                                                   |
| eps               | float  | 1.0 与大于 1.0 的最小可表示浮点数之间的差值。                                        |
| max               | float  | 最大可表示数。                                                                       |
| min               | float  | 最小可表示数（通常为 ``-max``）。                                                    |
| tiny              | float  | 最小的正规格化数。等同于 ``smallest_normal``。                                       |
| smallest_normal   | float  | 最小的正规格化数。参见注释。                                                         |
| resolution        | float  | 该类型的近似十进制分辨率，即 ``10**-precision``。                                    |

```{note}
  `torch.finfo` 的构造函数可以在不提供参数的情况下调用，此时会为 PyTorch 的默认 dtype（由 `torch.get_default_dtype` 返回）创建该类。
```

```{note}
  `smallest_normal` 返回最小的*规格化*数，但存在更小的次规格化数。更多信息请参见 https://en.wikipedia.org/wiki/Denormal_number。
```


## torch.iinfo


`torch.iinfo` 是一个表示整型 `torch.dtype`（例如 ``torch.uint8``、``torch.int8``、``torch.int16``、``torch.int32`` 和 ``torch.int64``）数值属性的对象。这类似于 [numpy.iinfo](https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html)。

一个 `torch.iinfo` 提供以下属性：

| 名称 | 类型 | 描述                              |
| :--- | :--- | :-------------------------------- |
| bits | int  | 该类型占用的位数。                |
| max  | int  | 最大可表示数。                    |
| min  | int  | 最小可表示数。                    |