// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  // Step 1. Split the reduction to get meatier (size(red) / 2)-way parallelism.
  // ===========================================================================
  %0 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %leading_eltwise, %reduction, %trailing_eltwise = transform.split_handle %0
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  %init_or_alloc_op, %more_parallel_fill_op, %more_parallel_op, %combiner_op =
    transform.structured.split_reduction %reduction
      { split_factor = 2, insert_split_dimension = 1 }
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  // Step 2. First level of tiling + fusion parallelizes to blocks. Tile the
  // trailing elementwise the same way we want to tile the reduction.
  // ===========================================================================
  %trailing_eltwise_grid_op, %grid_loop =
    transform.structured.tile_using_forall %trailing_eltwise tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Step 2.1: Cannot fuse across the "expand_shape" produced by reduction
  // splitting above, so we need to bubble that up via patterns and rematch
  // the entire structure.
  // TODO: bubbling should be a proper transform op, at which point we will be
  // able to preserve the handles.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.iree.bubble_expand
  } : !transform.any_op
  %fills = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %fill_2, %more_parallel_fill_2 = transform.split_handle %fill
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %expanded_eltwise, %more_parallel_2, %combiner_2, %trailing_eltwise_2 =
    transform.split_handle %generics
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  %forall_grid_2 = transform.structured.match ops{["scf.forall"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %not_trailing = transform.merge_handles %fill_2, %more_parallel_fill_2,
    %more_parallel_2, %expanded_eltwise, %combiner_2 : !transform.any_op
  transform.structured.fuse_into_containing_op %not_trailing into %forall_grid_2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Step 3. Second level of tiling + fusion parallelizes to threads. Also
  // fuse in the leading and trailing elementwise.
  // ===========================================================================
  %fill_1d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1xf32> in %variant_op : (!transform.any_op) -> !transform.any_op
  %block_trailing_eltwise_op, %forall_trailing_eltwise_op =
    transform.structured.tile_using_forall %trailing_eltwise_2 tile_sizes [1]
    ( mapping = [#gpu.thread<z>] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %block_combiner_op = transform.structured.match ops{["linalg.generic"]}
    attributes {iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %fill_and_reduction = transform.merge_handles %fill_1d, %block_combiner_op : !transform.any_op
  transform.structured.fuse_into_containing_op %fill_and_reduction into %forall_trailing_eltwise_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  %fill_2d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1x2xf32> in %variant_op : (!transform.any_op) -> !transform.any_op
  %grid_more_parallel_op = transform.structured.match ops{["linalg.generic"]}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %grid_eltwise_op = transform.structured.match ops{["linalg.generic"]}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %block_more_parallel_op, %forall_block_more_parallel_op =
    transform.structured.tile_using_forall %grid_more_parallel_op tile_sizes [1, 1]
    ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.fuse_into_containing_op %fill_2d into %forall_block_more_parallel_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.fuse_into_containing_op %grid_eltwise_op into %forall_block_more_parallel_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Step 4. Rank-reduce and vectorize.
  // ===========================================================================
  %func_1 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func_1 {
    transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    transform.apply_patterns.vector.cast_away_vector_leading_one_dim
  } : !transform.any_op
  %func_2 = transform.structured.vectorize_children_and_apply_patterns %func_1 : (!transform.any_op) -> !transform.any_op

  // Step 5. Bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %variant_op_2 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> !transform.any_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op

  // Step 6. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_3 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
  transform.iree.forall_to_workgroup %func_3 : (!transform.any_op) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_3 workgroup_dims = [32, 2, 1] : (!transform.any_op) -> ()

  // Step 7. Post-bufferization vector distribution with rank-reduction.
  // ===========================================================================
  transform.apply_patterns to %func_3 {
    transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    transform.apply_patterns.memref.fold_memref_alias_ops
    transform.apply_patterns.vector.cast_away_vector_leading_one_dim
  } : !transform.any_op
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
  // Don't complain about unsupported if (threadIdx.x == 0 && threadIdx.y == 0)
  // at this point.
  transform.sequence %variant_op_2 : !transform.any_op failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
    : (!transform.any_op) -> !transform.any_op
  }
  transform.iree.vector.warp_distribute %func_3 : (!transform.any_op) -> ()

  // Annotate the exported function as already translated.
  %exports = transform.structured.match ops{["hal.executable.export"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
  %none = transform.param.constant #iree_codegen.translation_info<None> -> !transform.any_param
  transform.annotate %exports "translation_info" = %none : !transform.any_op, !transform.any_param
}
