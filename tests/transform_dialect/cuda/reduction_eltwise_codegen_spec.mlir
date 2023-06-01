// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  // Step 1. Split the reduction to get meatier (size(red) / 2)-way parallelism.
  // ===========================================================================
  %0 = transform.structured.match ops{["linalg.generic"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op
  %reduction, %eltwise = transform.split_handle %0
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %init_or_alloc_op, %more_parallel_fill_op, %more_parallel_op, %combiner_op =
    transform.structured.split_reduction %reduction
      { split_factor = 2, insert_split_dimension = 1 }
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  // Canonicalizations.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()

  // Step 2. First level of tiling + fusion parallelizes to blocks. Tile the
  // trailing elementwise the same way we want to tile the reduction.
  // ===========================================================================
  %grid_loop, %eltwise_grid_op = transform.structured.tile_to_forall_op %eltwise
    tile_sizes [1] (mapping = [#gpu.block<x>])
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %grid_loop : (!transform.any_op) -> ()
  %not_eltwise = transform.merge_handles %fill, %more_parallel_fill_op, %more_parallel_op, %combiner_op
    : !transform.any_op
  transform.structured.fuse_into_containing_op %not_eltwise into %grid_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Canonicalizations.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()

  // Step 3. Second level of tiling + fusion parallelizes to threads.
  // ===========================================================================
  %fill_1d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1xf32> in %variant_op
    : (!transform.any_op) -> !transform.any_op
  %eltwise_block_loop, %eltwise_block_op =
    transform.structured.tile_to_forall_op %eltwise_grid_op tile_sizes [1]
    ( mapping = [#gpu.thread<z>] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %block_combiner_op = transform.structured.match ops{["linalg.generic"]}
    attributes {iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
  %combined_and_fill = transform.merge_handles %fill_1d, %block_combiner_op : !transform.any_op
  transform.structured.fuse_into_containing_op %combined_and_fill into %eltwise_block_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Canonicalizations.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()

  %fill_2d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1x2xf32> in %variant_op
    : (!transform.any_op) -> !transform.any_op
  %grid_more_parallel_op = transform.structured.match ops{["linalg.generic"]}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
  %forall_block_more_parallel_op, %block_more_parallel_op =
    transform.structured.tile_to_forall_op %grid_more_parallel_op tile_sizes [1, 1]
    ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.fuse_into_containing_op %fill_2d into %forall_block_more_parallel_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Canonicalizations.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()

  // Step 4. Rank-reduce and vectorize.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!transform.any_op) -> ()
  %func_3 = transform.structured.vectorize %func : (!transform.any_op) -> !transform.any_op

  // Step 5. Bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  transform.iree.apply_patterns %func_3 { fold_reassociative_reshapes } : (!transform.any_op) -> ()
  transform.iree.eliminate_empty_tensors %variant_op: (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3
    : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func: (!transform.any_op) -> ()

  // Step 6. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_5 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.forall_to_workgroup %func_5 : (!transform.any_op) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_5
      workgroup_dims = [32, 2, 1] : (!transform.any_op) -> ()

  // Step 7. Post-bufferization vector distribution with rank-reduction.
  // ===========================================================================
  transform.iree.apply_patterns %func_5 { rank_reducing_linalg, rank_reducing_vector, fold_memref_aliases } : (!transform.any_op) -> ()
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_3
    : (!transform.any_op) -> !transform.any_op
  // Don't complain about unsupported if (threadIdx.x == 0 && threadIdx.y == 0)
  // at this point.
  transform.sequence %variant_op_3 : !transform.any_op failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
    : (!transform.any_op) -> !transform.any_op
  }
  transform.iree.vector.warp_distribute %func_5 : (!transform.any_op) -> ()


  // Late canonicalizations.
  transform.iree.apply_patterns %variant_op_3
    { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()
}
