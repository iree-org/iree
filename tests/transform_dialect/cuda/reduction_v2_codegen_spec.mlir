// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %reduction = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  // Step 1. First level of tiling + fusion parallelizes to blocks.
  // ===========================================================================
  %forall_grid, %grid_reduction =
    transform.structured.tile_to_forall_op %reduction tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!pdl.operation) -> ()
  transform.structured.fuse_into_containing_op %fill into %forall_grid

  // Step 2. Split the reduction to get meatier parallelism.
  // ===========================================================================
  %forall, %block_more_parallel_fill_op_2, %block_more_parallel_op_2, %block_combiner_op_2 = 
    transform.structured.tile_reduction_using_scf %grid_reduction by tile_sizes = [0, 128]
  %_1:2 =
    transform.structured.tile_to_forall_op %block_more_parallel_op_2 num_threads [0, 32]
    ( mapping = [#gpu.thread<x>] )

  // Step 3. Second level of tiling parallelizes to threads.
  // ===========================================================================
  // 1st op is [parallel, parallel], map it to threadIdx.x by 4.
  %_2:2 =
    transform.structured.tile_to_forall_op %block_more_parallel_fill_op_2 tile_sizes [0, 4]
    ( mapping = [#gpu.thread<x>] )
  // 2nd op is [parallel, reduction] of 1x128, map the 1-dim to threadIdx.y to
  // trigger mapping of the reduction to threadIdx.x via predication via `if (x==0)`.
  %_3:2 =
    transform.structured.tile_to_forall_op %block_combiner_op_2 tile_sizes [1] 
    ( mapping = [#gpu.thread<y>] )

  // Step 4. Rank-reduce and vectorize.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
  %func_3 = transform.structured.vectorize %func

  // Step 5. Bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  // Canonicalization/CSE is needed before bufferization otherwise unnecessary
  // allocs will be created.
  transform.iree.apply_patterns %func_3
    { fold_reassociative_reshapes, canonicalization, tiling_canonicalization, cse } : (!pdl.operation) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %func_5 = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_5 { erase_unnecessary_tensor_operands } : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!pdl.operation) -> (!pdl.operation)
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

  // Step 6. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.forall_to_workgroup %func_7 : (!pdl.operation) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_7
      workgroup_dims = [32, 1, 1] : (!pdl.operation) -> ()

  // Step 7. Post-bufferization vector distribution with rank-reduction.
  // ===========================================================================
  transform.iree.apply_patterns %func_7 { rank_reducing_linalg, rank_reducing_vector, fold_memref_aliases } : (!pdl.operation) -> ()
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
  transform.iree.vector.warp_distribute %func_7
    : (!pdl.operation) -> ()

  // Late canonicalizations to cleanup and pass the checks
  transform.iree.apply_patterns %func_7
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation) -> ()
}
