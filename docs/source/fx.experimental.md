# torch.fx.experimental

> 模块：`torch.fx.experimental`

:::{warning}
这些 API 是实验性的，可能会随时更改，恕不另行通知。
:::


## 类

| 类 | 说明 |
|------|------|
| `torch.fx.experimental.sym_node.DynamicInt` | — |

## torch.fx.experimental.sym_node


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.is_channels_last_contiguous_2d` | — |
| `torch.fx.experimental.is_channels_last_contiguous_3d` | — |
| `torch.fx.experimental.is_channels_last_strides_2d` | — |
| `torch.fx.experimental.is_channels_last_strides_3d` | — |
| `torch.fx.experimental.is_contiguous` | — |
| `torch.fx.experimental.is_non_overlapping_and_dense_indicator` | — |
| `torch.fx.experimental.method_to_operator` | — |
| `torch.fx.experimental.sympy_is_channels_last_contiguous_2d` | — |
| `torch.fx.experimental.sympy_is_channels_last_contiguous_3d` | — |
| `torch.fx.experimental.sympy_is_channels_last_strides_2d` | — |
| `torch.fx.experimental.sympy_is_channels_last_strides_3d` | — |
| `torch.fx.experimental.sympy_is_channels_last_strides_generic` | — |
| `torch.fx.experimental.sympy_is_contiguous` | — |
| `torch.fx.experimental.sympy_is_contiguous_generic` | — |
| `torch.fx.experimental.to_node` | — |

## torch.fx.experimental.symbolic_shapes


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.ShapeEnv` | — |
| `torch.fx.experimental.DimDynamic` | — |
| `torch.fx.experimental.StrictMinMaxConstraint` | — |
| `torch.fx.experimental.RelaxedUnspecConstraint` | — |
| `torch.fx.experimental.EqualityConstraint` | — |
| `torch.fx.experimental.SymbolicContext` | — |
| `torch.fx.experimental.StatelessSymbolicContext` | — |
| `torch.fx.experimental.StatefulSymbolicContext` | — |
| `torch.fx.experimental.SubclassSymbolicContext` | — |
| `torch.fx.experimental.DimConstraints` | — |
| `torch.fx.experimental.ShapeEnvSettings` | — |
| `torch.fx.experimental.ConvertIntKey` | — |
| `torch.fx.experimental.CallMethodKey` | — |
| `torch.fx.experimental.PropagateUnbackedSymInts` | — |
| `torch.fx.experimental.DivideByKey` | — |
| `torch.fx.experimental.InnerTensorKey` | — |
| `torch.fx.experimental.Specialization` | — |
| `torch.fx.experimental.is_concrete_int` | — |
| `torch.fx.experimental.is_concrete_bool` | — |
| `torch.fx.experimental.is_concrete_float` | — |
| `torch.fx.experimental.has_free_symbols` | — |
| `torch.fx.experimental.has_free_unbacked_symbols` | — |
| `torch.fx.experimental.guard_or_true` | — |
| `torch.fx.experimental.guard_or_false` | — |
| `torch.fx.experimental.guard_size_oblivious` | — |
| `torch.fx.experimental.sym_and` | — |
| `torch.fx.experimental.sym_eq` | — |
| `torch.fx.experimental.sym_or` | — |
| `torch.fx.experimental.constrain_range` | — |
| `torch.fx.experimental.constrain_unify` | — |
| `torch.fx.experimental.canonicalize_bool_expr` | — |
| `torch.fx.experimental.statically_known_true` | — |
| `torch.fx.experimental.statically_known_false` | — |
| `torch.fx.experimental.has_static_value` | — |
| `torch.fx.experimental.lru_cache` | — |
| `torch.fx.experimental.check_consistent` | — |
| `torch.fx.experimental.compute_unbacked_bindings` | — |
| `torch.fx.experimental.rebind_unbacked` | — |
| `torch.fx.experimental.resolve_unbacked_bindings` | — |
| `torch.fx.experimental.is_accessor_node` | — |
| `torch.fx.experimental.cast_symbool_to_symint_guardless` | — |
| `torch.fx.experimental.create_contiguous` | — |
| `torch.fx.experimental.error` | — |
| `torch.fx.experimental.eval_guards` | — |
| `torch.fx.experimental.eval_is_non_overlapping_and_dense` | — |
| `torch.fx.experimental.find_symbol_binding_fx_nodes` | — |
| `torch.fx.experimental.free_symbols` | — |
| `torch.fx.experimental.free_unbacked_symbols` | — |
| `torch.fx.experimental.fx_placeholder_targets` | — |
| `torch.fx.experimental.fx_placeholder_vals` | — |
| `torch.fx.experimental.guard_bool` | — |
| `torch.fx.experimental.guard_float` | — |
| `torch.fx.experimental.guard_int` | — |
| `torch.fx.experimental.guard_scalar` | — |
| `torch.fx.experimental.guarding_hint_or_throw` | — |
| `torch.fx.experimental.has_guarding_hint` | — |
| `torch.fx.experimental.has_symbolic_sizes_strides` | — |
| `torch.fx.experimental.is_nested_int` | — |
| `torch.fx.experimental.is_symbol_binding_fx_node` | — |
| `torch.fx.experimental.is_symbolic` | — |
| `torch.fx.experimental.optimization_hint` | — |
| `torch.fx.experimental.expect_true` | — |
| `torch.fx.experimental.log_lru_cache_stats` | — |

## torch.fx.experimental.proxy_tensor


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.make_fx` | — |
| `torch.fx.experimental.handle_sym_dispatch` | — |
| `torch.fx.experimental.get_proxy_mode` | — |
| `torch.fx.experimental.maybe_enable_thunkify` | — |
| `torch.fx.experimental.maybe_disable_thunkify` | — |
| `torch.fx.experimental.thunkify` | — |
| `torch.fx.experimental.track_tensor` | — |
| `torch.fx.experimental.track_tensor_tree` | — |
| `torch.fx.experimental.decompose` | — |
| `torch.fx.experimental.disable_autocast_cache` | — |
| `torch.fx.experimental.disable_proxy_modes_tracing` | — |
| `torch.fx.experimental.dispatch_trace` | — |
| `torch.fx.experimental.extract_val` | — |
| `torch.fx.experimental.fake_signature` | — |
| `torch.fx.experimental.fetch_object_proxy` | — |
| `torch.fx.experimental.fetch_sym_proxy` | — |
| `torch.fx.experimental.has_proxy_slot` | — |
| `torch.fx.experimental.is_sym_node` | — |
| `torch.fx.experimental.maybe_handle_decomp` | — |
| `torch.fx.experimental.proxy_call` | — |
| `torch.fx.experimental.set_meta` | — |
| `torch.fx.experimental.set_original_aten_op` | — |
| `torch.fx.experimental.set_proxy_slot` | — |
| `torch.fx.experimental.snapshot_fake` | — |

## torch.fx.experimental.optimization


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.extract_subgraph` | — |
| `torch.fx.experimental.matches_module_pattern` | — |
| `torch.fx.experimental.modules_to_mkldnn` | — |
| `torch.fx.experimental.optimize_for_inference` | — |
| `torch.fx.experimental.remove_dropout` | — |
| `torch.fx.experimental.replace_node_module` | — |
| `torch.fx.experimental.reset_modules` | — |
| `torch.fx.experimental.use_mkl_length` | — |

## torch.fx.experimental.recording


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.record_shapeenv_event` | — |
| `torch.fx.experimental.replay_shape_env_events` | — |
| `torch.fx.experimental.shape_env_check_state_equal` | — |

## torch.fx.experimental.unification.core


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.reify` | — |

## torch.fx.experimental.unification.multipledispatch.utils


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.typename` | — |
| `torch.fx.experimental.expand_tuples` | — |
| `torch.fx.experimental.groupby` | — |
| `torch.fx.experimental.raises` | — |
| `torch.fx.experimental.reverse_dict` | — |

## torch.fx.experimental.unification.unification_tools


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.assoc` | — |
| `torch.fx.experimental.assoc_in` | — |
| `torch.fx.experimental.dissoc` | — |
| `torch.fx.experimental.first` | — |
| `torch.fx.experimental.groupby` | — |
| `torch.fx.experimental.keyfilter` | — |
| `torch.fx.experimental.keymap` | — |
| `torch.fx.experimental.merge` | — |
| `torch.fx.experimental.merge_with` | — |
| `torch.fx.experimental.update_in` | — |
| `torch.fx.experimental.valfilter` | — |
| `torch.fx.experimental.valmap` | — |
| `torch.fx.experimental.itemfilter` | — |
| `torch.fx.experimental.itemmap` | — |

## torch.fx.experimental.migrate_gradual_types.transform_to_z3


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.transform_algebraic_expression` | — |
| `torch.fx.experimental.transform_all_constraints` | — |
| `torch.fx.experimental.transform_all_constraints_trace_time` | — |
| `torch.fx.experimental.transform_dimension` | — |
| `torch.fx.experimental.transform_to_z3` | — |
| `torch.fx.experimental.transform_var` | — |
| `torch.fx.experimental.evaluate_conditional_with_constraints` | — |

## torch.fx.experimental.migrate_gradual_types.constraint


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.is_algebraic_expression` | — |
| `torch.fx.experimental.is_bool_expr` | — |
| `torch.fx.experimental.is_dim` | — |

## torch.fx.experimental.migrate_gradual_types.constraint_generator


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.adaptive_inference_rule` | — |
| `torch.fx.experimental.assert_inference_rule` | — |
| `torch.fx.experimental.batchnorm_inference_rule` | — |
| `torch.fx.experimental.bmm_inference_rule` | — |
| `torch.fx.experimental.embedding_inference_rule` | — |
| `torch.fx.experimental.embedding_inference_rule_functional` | — |
| `torch.fx.experimental.eq_inference_rule` | — |
| `torch.fx.experimental.equality_inference_rule` | — |
| `torch.fx.experimental.expand_inference_rule` | — |
| `torch.fx.experimental.full_inference_rule` | — |
| `torch.fx.experimental.gt_inference_rule` | — |
| `torch.fx.experimental.lt_inference_rule` | — |
| `torch.fx.experimental.masked_fill_inference_rule` | — |
| `torch.fx.experimental.neq_inference_rule` | — |
| `torch.fx.experimental.tensor_inference_rule` | — |
| `torch.fx.experimental.torch_dim_inference_rule` | — |
| `torch.fx.experimental.torch_linear_inference_rule` | — |
| `torch.fx.experimental.type_inference_rule` | — |
| `torch.fx.experimental.view_inference_rule` | — |
| `torch.fx.experimental.register_inference_rule` | — |
| `torch.fx.experimental.transpose_inference_rule` | — |

## torch.fx.experimental.migrate_gradual_types.constraint_transformation


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.apply_padding` | — |
| `torch.fx.experimental.calc_last_two_dims` | — |
| `torch.fx.experimental.create_equality_constraints_for_broadcasting` | — |
| `torch.fx.experimental.is_target_div_by_dim` | — |
| `torch.fx.experimental.no_broadcast_dim_with_index` | — |
| `torch.fx.experimental.register_transformation_rule` | — |
| `torch.fx.experimental.transform_constraint` | — |
| `torch.fx.experimental.transform_get_item` | — |
| `torch.fx.experimental.transform_get_item_tensor` | — |
| `torch.fx.experimental.transform_index_select` | — |
| `torch.fx.experimental.transform_transpose` | — |
| `torch.fx.experimental.valid_index` | — |
| `torch.fx.experimental.valid_index_tensor` | — |
| `torch.fx.experimental.is_dim_div_by_target` | — |

## torch.fx.experimental.graph_gradual_typechecker


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.adaptiveavgpool2d_check` | — |
| `torch.fx.experimental.adaptiveavgpool2d_inference_rule` | — |
| `torch.fx.experimental.all_eq` | — |
| `torch.fx.experimental.bn2d_inference_rule` | — |
| `torch.fx.experimental.calculate_out_dimension` | — |
| `torch.fx.experimental.conv_refinement_rule` | — |
| `torch.fx.experimental.conv_rule` | — |
| `torch.fx.experimental.element_wise_eq` | — |
| `torch.fx.experimental.expand_to_tensor_dim` | — |
| `torch.fx.experimental.first_two_eq` | — |
| `torch.fx.experimental.register_algebraic_expressions_inference_rule` | — |
| `torch.fx.experimental.register_inference_rule` | — |
| `torch.fx.experimental.register_refinement_rule` | — |
| `torch.fx.experimental.transpose_inference_rule` | — |

## torch.fx.experimental.meta_tracer


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.embedding_override` | — |
| `torch.fx.experimental.functional_relu_override` | — |
| `torch.fx.experimental.nn_layernorm_override` | — |
| `torch.fx.experimental.proxys_to_metas` | — |
| `torch.fx.experimental.symbolic_trace` | — |
| `torch.fx.experimental.torch_abs_override` | — |
| `torch.fx.experimental.torch_nn_relu_override` | — |
| `torch.fx.experimental.torch_relu_override` | — |
| `torch.fx.experimental.torch_where_override` | — |

## torch.fx.experimental.accelerator_partitioner


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.check_dependency` | — |
| `torch.fx.experimental.combine_two_partitions` | — |
| `torch.fx.experimental.reorganize_partitions` | — |
| `torch.fx.experimental.reset_partition_device` | — |
| `torch.fx.experimental.set_parents_and_children` | — |

## torch.fx.experimental.debug


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.set_trace` | — |

## torch.fx.experimental.merge_matmul


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.are_nodes_independent` | — |
| `torch.fx.experimental.may_depend_on` | — |
| `torch.fx.experimental.merge_matmul` | — |

## torch.fx.experimental.unification.match


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.edge` | — |
| `torch.fx.experimental.match` | — |
| `torch.fx.experimental.ordering` | — |
| `torch.fx.experimental.supercedes` | — |

## torch.fx.experimental.unification.more


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.reify_object` | — |
| `torch.fx.experimental.unifiable` | — |
| `torch.fx.experimental.unify_object` | — |

## torch.fx.experimental.unification.multipledispatch.conflict


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.ambiguities` | — |
| `torch.fx.experimental.ambiguous` | — |
| `torch.fx.experimental.consistent` | — |
| `torch.fx.experimental.edge` | — |
| `torch.fx.experimental.ordering` | — |
| `torch.fx.experimental.super_signature` | — |
| `torch.fx.experimental.supercedes` | — |

## torch.fx.experimental.unification.multipledispatch.core


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.dispatch` | — |
| `torch.fx.experimental.ismethod` | — |

## torch.fx.experimental.unification.multipledispatch.dispatcher


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.ambiguity_warn` | — |
| `torch.fx.experimental.halt_ordering` | — |
| `torch.fx.experimental.restart_ordering` | — |
| `torch.fx.experimental.source` | — |
| `torch.fx.experimental.str_signature` | — |
| `torch.fx.experimental.variadic_signature_matches` | — |
| `torch.fx.experimental.variadic_signature_matches_iter` | — |
| `torch.fx.experimental.warning_text` | — |

## torch.fx.experimental.unification.multipledispatch.variadic


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.isvariadic` | — |

## torch.fx.experimental.unification.utils


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.freeze` | — |
| `torch.fx.experimental.hashable` | — |
| `torch.fx.experimental.raises` | — |
| `torch.fx.experimental.reverse_dict` | — |
| `torch.fx.experimental.transitive_get` | — |
| `torch.fx.experimental.xfail` | — |

## torch.fx.experimental.unification.variable


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.var` | — |
| `torch.fx.experimental.variables` | — |
| `torch.fx.experimental.vars` | — |

## torch.fx.experimental.unify_refinements


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.check_for_type_equality` | — |
| `torch.fx.experimental.infer_symbolic_types` | — |
| `torch.fx.experimental.infer_symbolic_types_single_pass` | — |
| `torch.fx.experimental.substitute_all_types` | — |
| `torch.fx.experimental.substitute_solution_one_type` | — |
| `torch.fx.experimental.unify_eq` | — |

## torch.fx.experimental.validator


## 函数

| 函数 | 说明 |
|------|------|
| `torch.fx.experimental.bisect` | — |
| `torch.fx.experimental.translation_validation_enabled` | — |
| `torch.fx.experimental.translation_validation_timeout` | — |
| `torch.fx.experimental.z3op` | — |
| `torch.fx.experimental.z3str` | — |
