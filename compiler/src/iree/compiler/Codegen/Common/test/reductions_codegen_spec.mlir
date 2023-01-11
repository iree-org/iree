// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.iree.register_match_callbacks

  %_, %fill, %reduction, %maybe_trailing_0 =
    transform.iree.match_callback failures(propagate) "reduction"(%arg0)
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  
  %0, %1, %2, %combiner_op =
    transform.structured.split_reduction %reduction { split_factor = 2, insert_split_dimension = 1 }

  %fusion_root_1, %fusion_group_1 = transform.iree.take_first %maybe_trailing_0, %combiner_op
    : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
  transform.structured.tile_to_foreach_thread_op %fusion_root_1 tile_sizes [1]
    ( mapping = [#gpu.block<x>] )
  
  // TODO: bubbling should be a proper transform op, at which point we will be
  // able to preserve the handles and avoid rematching below.
  %func = transform.structured.match ops{["func.func"]} in %arg0
  %func_1 = transform.iree.apply_patterns %func { bubble_collapse_expand }

  %maybe_leading, %original_fill, %more_parallel_fill, %parallel_reduction, %combiner_reduction, %maybe_trailing =
    transform.iree.match_callback failures(propagate) "split_reduction"(%arg0)
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

  %fusion_root_1_updated, %fusion_group_1_updated =
    transform.iree.take_first %maybe_trailing, %combiner_reduction
    : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
  // TODO: we need an extra navigation op similar to get_parent_loop to avoid rematching.
  // %grid_loop = transform.iree.get_parent_of_type("scf.foreach_thread") %fusion_root_1_updated 
  %grid_loop = transform.structured.match ops{["scf.foreach_thread"]} in %arg0
  %fusion_group_1_full = transform.merge_handles %fusion_group_1_updated,
    %maybe_leading, %original_fill, %more_parallel_fill, %parallel_reduction
    : !pdl.operation
  transform.structured.fuse_into_containing_op %fusion_group_1_full into %grid_loop

  // Step 3.
  %maybe_leading_2, %original_fill_2, %more_parallel_fill_2, %parallel_reduction_2, %combiner_reduction_2, %maybe_trailing_2 =
    transform.iree.match_callback failures(propagate) "split_reduction"(%arg0)
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %fusion_root_22, %fusion_group_22 =
    transform.iree.take_first %maybe_trailing_2, %combiner_reduction_2
    : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fusion_group_22_full = transform.merge_handles %fusion_group_22, %original_fill_2
    : !pdl.operation
  %block_loop_22, %fusion_root_22_tiled =
    transform.structured.tile_to_foreach_thread_op %fusion_root_22
    tile_sizes [1] ( mapping = [#gpu.thread<z>] )
  transform.structured.fuse_into_containing_op %fusion_group_22_full into %block_loop_22

  %fusion_group_21 = transform.merge_handles %maybe_leading_2, %more_parallel_fill_2
    : !pdl.operation
  %block_loop_21, %fusion_root_21_tiled =
    transform.structured.tile_to_foreach_thread_op %parallel_reduction_2
    tile_sizes [1, 1] ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
  transform.structured.fuse_into_containing_op %fusion_group_21 into %block_loop_21
  
  // Step 4. Rank-reduce.
  // ===========================================================================
  %func_2 = transform.iree.apply_patterns %func_1 { rank_reducing }

  // We don't perform any following transformation (vectorization, bufferizaton,
  // mapping) because this schedule is applied to Linalg-only code without the
  // surrounding context and because it would make it difficult to detect, e.g.,
  // lack of fusion.
}
