// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %reduction = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  // Step 1. First level of tiling + fusion parallelizes to blocks.
  // ===========================================================================
  %forall_grid, %grid_reduction =
    transform.structured.tile_to_forall_op %reduction tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()
  transform.structured.fuse_into_containing_op %fill into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Step 2. Split the reduction to get meatier parallelism.
  // ===========================================================================
  %forall, %block_more_parallel_fill_op_2, %block_more_parallel_op_2, %block_combiner_op_2 = 
    transform.structured.tile_reduction_using_scf %grid_reduction by tile_sizes = [0, 128]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  %_1:2 =
    transform.structured.tile_to_forall_op %block_more_parallel_op_2 num_threads [0, 32]
    ( mapping = [#gpu.thread<x>] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Step 3. Second level of tiling parallelizes to threads.
  // ===========================================================================
  // 1st op is [parallel, parallel], map it to threadIdx.x by 4.
  %_2:2 =
    transform.structured.tile_to_forall_op %block_more_parallel_fill_op_2 tile_sizes [0, 4]
    ( mapping = [#gpu.thread<x>] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  // 2nd op is [parallel, reduction] of 1x128, map the 1-dim to threadIdx.y to
  // trigger mapping of the reduction to threadIdx.x via predication via `if (x==0)`.
  %_3:2 =
    transform.structured.tile_to_forall_op %block_combiner_op_2 tile_sizes [1] 
    ( mapping = [#gpu.thread<y>] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Step 4. Rank-reduce and vectorize.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!transform.any_op) -> ()
  %func_3 = transform.structured.vectorize %func : (!transform.any_op) -> !transform.any_op

  // Step 5. Bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  // Canonicalization/CSE is needed before bufferization otherwise unnecessary
  // allocs will be created.
  transform.iree.apply_patterns %func_3
    { fold_reassociative_reshapes, canonicalization, tiling_canonicalization, cse } : (!transform.any_op) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %func_5 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %func_5 { erase_unnecessary_tensor_operands } : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()

  // Step 6. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_7
      workgroup_dims = [32, 1, 1] : (!transform.any_op) -> ()

  // Step 7. Post-bufferization vector distribution with rank-reduction.
  // ===========================================================================
  transform.iree.apply_patterns %func_7 { rank_reducing_linalg, rank_reducing_vector, fold_memref_aliases } : (!transform.any_op) -> ()
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_3 
    : (!transform.any_op) -> !transform.any_op
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 } : (!transform.any_op) -> !transform.any_op
  transform.iree.vector.warp_distribute %func_7
    : (!transform.any_op) -> ()

  // Late canonicalizations to cleanup and pass the checks
  transform.iree.apply_patterns %func_7
    { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()
}
