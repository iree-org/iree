// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.iree.register_match_callbacks

  %maybe_leading, %original_fill, %reduction, %maybe_trailing_0 =
    transform.iree.match_callback failures(propagate) "reduction"(%arg0)
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  
  %_, %more_parallel_fill, %parallel_reduction, %combiner_op =
    transform.structured.split_reduction %reduction { split_factor = 2, insert_split_dimension = 1 }

  // Step 1. Map to a single block by tiling with size 1 and fusing.
  %fusion_root_1, %fusion_group_1 = transform.iree.take_first %maybe_trailing_0, %combiner_op
    : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
  %grid_loop, %outer_tiled = transform.structured.tile_to_forall_op %fusion_root_1 tile_sizes [1]
    ( mapping = [#gpu.block<x>] )
  
  %func = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %func_1 = transform.iree.apply_patterns %func { bubble_collapse_expand }

  // Excessively eager canonicalization results in `fill`s being "fused" due to
  // swapping with `extract_slice`, which confuses the fusion operation below.
  // Wrap fusion into a non-canonicalized sequence.
  %fused_2, %parallel_reduction_2, %more_parallel_fill_2, %original_fill_2, %maybe_leading_2 =
    transform.sequence %arg0 : !pdl.operation -> !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation
    failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %fused_22 = transform.structured.fuse_into_containing_op %fusion_group_1 into %grid_loop
    %parallel_reduction_22 = transform.structured.fuse_into_containing_op %parallel_reduction into %grid_loop
    %more_parallel_fill_22 = transform.structured.fuse_into_containing_op %more_parallel_fill into %grid_loop
    %original_fill_22 = transform.structured.fuse_into_containing_op %original_fill into %grid_loop
    %maybe_leading_22 = transform.structured.fuse_into_containing_op %maybe_leading into %grid_loop

    transform.yield %fused_22, %parallel_reduction_22, %more_parallel_fill_22, %original_fill_22, %maybe_leading_22
      : !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation
  }

  // Step 2. Map reduction to thread X and parallel dimension to other threads.
  // ===========================================================================
  %fusion_group_22_full = transform.merge_handles %fused_2, %original_fill_2
    : !pdl.operation
  %block_loop_22, %fusion_root_22_tiled =
    transform.structured.tile_to_forall_op %outer_tiled
    tile_sizes [1] ( mapping = [#gpu.thread<z>] )
  transform.structured.fuse_into_containing_op %fusion_group_22_full into %block_loop_22

  %fusion_group_21 = transform.merge_handles %maybe_leading_2, %more_parallel_fill_2
    : !pdl.operation
  %block_loop_21, %fusion_root_21_tiled =
    transform.structured.tile_to_forall_op %parallel_reduction_2
    tile_sizes [1, 1] ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
  transform.structured.fuse_into_containing_op %fusion_group_21 into %block_loop_21
  
  // Step 3. Rank-reduce.
  // ===========================================================================
  %func_2 = transform.iree.apply_patterns %func_1 {  rank_reducing_linalg, rank_reducing_vector }

  // We don't perform any following transformation (vectorization, bufferizaton,
  // mapping) because this schedule is applied to Linalg-only code without the
  // surrounding context and because it would make it difficult to detect, e.g.,
  // lack of fusion.
}
