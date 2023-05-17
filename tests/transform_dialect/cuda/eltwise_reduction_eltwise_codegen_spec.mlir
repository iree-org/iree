// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  // Step 1. Split the reduction to get meatier (size(red) / 2)-way parallelism.
  // ===========================================================================
  %0 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %leading_eltwise, %reduction, %trailing_eltwise = transform.split_handle %0
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  %init_or_alloc_op, %more_parallel_fill_op, %more_parallel_op, %combiner_op =
    transform.structured.split_reduction %reduction
      { split_factor = 2, insert_split_dimension = 1 }
       : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

  // Step 2. First level of tiling + fusion parallelizes to blocks. Tile the
  // trailing elementwise the same way we want to tile the reduction.
  // ===========================================================================
  %grid_loop, %trailing_eltwise_grid_op =
    transform.structured.tile_to_forall_op %trailing_eltwise tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

  // Step 2.1: Cannot fuse across the "expand_shape" produced by reduction
  // splitting above, so we need to bubble that up via patterns and rematch
  // the entire structure.
  // TODO: bubbling should be a proper transform op, at which point we will be
  // able to preserve the handles.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func { bubble_expand } : (!pdl.operation) -> ()
  %fills = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %fill_2, %more_parallel_fill_2 = transform.split_handle %fill
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %expanded_eltwise, %more_parallel_2, %combiner_2, %trailing_eltwise_2 =
    transform.split_handle %generics
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %forall_grid_2 = transform.structured.match ops{["scf.forall"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %not_trailing = transform.merge_handles %fill_2, %more_parallel_fill_2,
    %more_parallel_2, %expanded_eltwise, %combiner_2 : !pdl.operation
  transform.structured.fuse_into_containing_op %not_trailing into %forall_grid_2 : (!pdl.operation, !pdl.operation) -> !pdl.operation

  // Step 3. Second level of tiling + fusion parallelizes to threads. Also
  // fuse in the leading and trailing elementwise.
  // ===========================================================================
  %fill_1d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1xf32> in %variant_op : (!pdl.operation) -> !pdl.operation
  %forall_trailing_eltwise_op, %block_trailing_eltwise_op =
    transform.structured.tile_to_forall_op %trailing_eltwise_2 tile_sizes [1]
    ( mapping = [#gpu.thread<z>] )
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %block_combiner_op = transform.structured.match ops{["linalg.generic"]}
    attributes {iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %fill_and_reduction = transform.merge_handles %fill_1d, %block_combiner_op : !pdl.operation
  transform.structured.fuse_into_containing_op %fill_and_reduction into %forall_trailing_eltwise_op : (!pdl.operation, !pdl.operation) -> !pdl.operation

  %fill_2d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1x2xf32> in %variant_op : (!pdl.operation) -> !pdl.operation
  %grid_more_parallel_op = transform.structured.match ops{["linalg.generic"]}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %grid_eltwise_op = transform.structured.match ops{["linalg.generic"]}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %forall_block_more_parallel_op, %block_more_parallel_op =
    transform.structured.tile_to_forall_op %grid_more_parallel_op tile_sizes [1, 1]
    ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  transform.structured.fuse_into_containing_op %fill_2d into %forall_block_more_parallel_op : (!pdl.operation, !pdl.operation) -> !pdl.operation
  transform.structured.fuse_into_containing_op %grid_eltwise_op into %forall_block_more_parallel_op : (!pdl.operation, !pdl.operation) -> !pdl.operation

  // Step 4. Rank-reduce and vectorize.
  // ===========================================================================
  %func_1 = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_1 {  rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
  %func_2 = transform.structured.vectorize %func_1 : (!pdl.operation) -> !pdl.operation

  // Step 5. Bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %variant_op_2 = transform.iree.bufferize { target_gpu } %variant_op : (!pdl.operation) -> !pdl.operation
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

  // Step 6. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_3 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!pdl.operation) -> !pdl.operation
  transform.iree.forall_to_workgroup %func_3 : (!pdl.operation) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_3 workgroup_dims = [32, 2, 1] : (!pdl.operation) -> ()

  // Step 7. Post-bufferization vector distribution with rank-reduction.
  // ===========================================================================
  transform.iree.apply_patterns %func_3 { rank_reducing_linalg, rank_reducing_vector, fold_memref_aliases } : (!pdl.operation) -> ()
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_2 : (!pdl.operation) -> !pdl.operation
  // Don't complain about unsupported if (threadIdx.x == 0 && threadIdx.y == 0)
  // at this point.
  transform.sequence %variant_op_2 : !pdl.operation failures(suppress) {
  ^bb0(%arg0: !pdl.operation):
    transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
  }
  transform.iree.vector.warp_distribute %func_3 : (!pdl.operation) -> ()
}
