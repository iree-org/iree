// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %reduction = transform.structured.match ops{["linalg.generic"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation

  // Step 1. First level of tiling + fusion parallelizes to blocks.
  // ===========================================================================
  %forall_grid, %grid_reduction =
    transform.iree.tile_to_forall_and_workgroup_count_region %reduction tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
  transform.structured.fuse_into_containing_op %fill into %forall_grid

  // Canonicalizations.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation) -> ()

  // Step 2. Split the reduction to get meatier parallelism.
  // This also parallelizes to threads.
  // ===========================================================================
  %forall, %block_more_parallel_fill_op_2, %block_more_parallel_op_2, %block_combiner_op_2 = 
     transform.structured.tile_reduction_using_forall %grid_reduction 
        by num_threads = [0, 1024], tile_sizes = [0, 1], mapping = [#gpu.thread<x>]

  // Fuse the fill and pointwise to privatize them.
  transform.structured.fuse_into_containing_op %block_more_parallel_fill_op_2
    into %forall

  // block_combiner_op_2 op is [parallel, reduction] of 1x384 that cannot fuse.
  // map the 1-dim to threadIdx.y to trigger mapping of the reduction to 
  // threadIdx.x via predication via `if (x==0)`.
  transform.structured.tile_to_forall_op %block_combiner_op_2 num_threads [1] 
    ( mapping = [#gpu.thread<y>] )

  // Canonicalizations.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation) -> ()

  // Step 3. Rank-reduce and vectorize.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  // TODO: masked vectorization on block_more_parallel_op_2 if we want 
  // vector<4> to work as intended.
  transform.iree.apply_patterns %func 
    { rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
  %func_3 = transform.structured.vectorize %func

  // Canonicalizations is necessary to get rid of some tensor.cast that block
  // hoisting.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation) -> ()
  %func_4 = transform.structured.hoist_redundant_tensor_subsets %func_3
    : (!pdl.operation) -> !pdl.operation


  // Step 4. Bufferize and drop HAL descriptor from memref ops.
  // ===========================================================================
  // Canonicalizations required before bufferization to avoid unnecessary allocs.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_4 { fold_reassociative_reshapes } : (!pdl.operation) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %func_6 = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_6 { erase_unnecessary_tensor_operands } : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op
    : (!pdl.operation) -> !pdl.operation
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

  // Step 5. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.forall_to_workgroup %func_m : (!pdl.operation) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_m
      workgroup_dims = [1024, 1, 1] : (!pdl.operation) -> ()

  // Step 6. Post-bufferization vector distribution with rank-reduction.
  // ===========================================================================
  transform.iree.apply_patterns %func_m { rank_reducing_linalg, rank_reducing_vector, fold_memref_aliases } : (!pdl.operation) -> ()
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
  transform.iree.vector.warp_distribute %func_m
    : (!pdl.operation) -> ()

  // Late canonicalizations.
  transform.iree.apply_patterns %variant_op_3
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation) -> ()
}
