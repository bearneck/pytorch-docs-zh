# torch.xpu

> 模块：`torch.xpu`


## 函数

| 函数 | 说明 |
|------|------|
| `torch.xpu.StreamContext` | — |
| `torch.xpu.can_device_access_peer` | — |
| `torch.xpu.current_device` | — |
| `torch.xpu.current_stream` | — |
| `torch.xpu.device` | — |
| `torch.xpu.device_count` | — |
| `torch.xpu.device_of` | — |
| `torch.xpu.get_arch_list` | — |
| `torch.xpu.get_device_capability` | — |
| `torch.xpu.get_device_name` | — |
| `torch.xpu.get_device_properties` | — |
| `torch.xpu.get_gencode_flags` | — |
| `torch.xpu.get_stream_from_external` | — |
| `torch.xpu.init` | — |
| `torch.xpu.is_available` | — |
| `torch.xpu.is_bf16_supported` | — |
| `torch.xpu.is_initialized` | — |
| `torch.xpu.is_tf32_supported` | — |
| `torch.xpu.set_device` | — |
| `torch.xpu.set_stream` | — |
| `torch.xpu.stream` | — |
| `torch.xpu.synchronize` | — |

## 随机数生成器


## 函数

| 函数 | 说明 |
|------|------|
| `torch.xpu.get_rng_state` | — |
| `torch.xpu.get_rng_state_all` | — |
| `torch.xpu.initial_seed` | — |
| `torch.xpu.manual_seed` | — |
| `torch.xpu.manual_seed_all` | — |
| `torch.xpu.seed` | — |
| `torch.xpu.seed_all` | — |
| `torch.xpu.set_rng_state` | — |
| `torch.xpu.set_rng_state_all` | — |

## 流和事件


## 函数

| 函数 | 说明 |
|------|------|
| `torch.xpu.Event` | — |
| `torch.xpu.Stream` | — |

## 图


## 函数

| 函数 | 说明 |
|------|------|
| `torch.xpu.is_current_stream_capturing` | — |
| `torch.xpu.graph_pool_handle` | — |
| `torch.xpu.XPUGraph` | — |
| `torch.xpu.graph` | — |
| `torch.xpu.make_graphed_callables` | — |

## 内存管理


## 函数

| 函数 | 说明 |
|------|------|
| `torch.xpu.XPUPluggableAllocator` | — |
| `torch.xpu.change_current_allocator` | — |
| `torch.xpu.empty_cache` | — |
| `torch.xpu.get_per_process_memory_fraction` | — |
| `torch.xpu.max_memory_allocated` | — |
| `torch.xpu.max_memory_reserved` | — |
| `torch.xpu.mem_get_info` | — |
| `torch.xpu.memory_allocated` | — |
| `torch.xpu.memory_reserved` | — |
| `torch.xpu.memory_snapshot` | — |
| `torch.xpu.memory_stats` | — |
| `torch.xpu.memory_stats_as_nested_dict` | — |
| `torch.xpu.reset_accumulated_memory_stats` | — |
| `torch.xpu.reset_peak_memory_stats` | — |
| `torch.xpu.set_per_process_memory_fraction` | — |
| `torch.xpu.MemPool` | — |

## 类

| 类 | 说明 |
|------|------|
| `torch.xpu.use_mem_pool` | — |
