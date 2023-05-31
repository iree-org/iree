// RUN: iree-opt %s

// Codegen
transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):

  // Step 1. First level of tiling + fusion parallelizes to blocks.
  // ==============================================================
  %root = transform.structured.match interface{LinalgOp}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %red = transform.structured.match interface{LinalgOp}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %not_root = merge_handles %fill, %red : !pdl.operation
  %forall, %tiled_generic =
    transform.structured.tile_to_forall_op %root tile_sizes [1, 4]
    ( mapping = [#gpu.block<x>, #gpu.block<y>] )
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall : (!pdl.operation) -> ()
  transform.structured.fuse_into_containing_op %not_root into %forall : (!pdl.operation, !pdl.operation) -> !pdl.operation

  // Step 2. Second level of tiling + fusion parallelizes to threads.
  // ================================================================
  %fill_linalg = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %reduction_linalg = transform.structured.match ops{["linalg.generic"]}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %parallel_linalg = transform.structured.match ops{["linalg.generic"]}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %forall_reduction, %tiled_reduction_generic =
    transform.structured.tile_to_forall_op %reduction_linalg tile_sizes [1, 1]
      ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  // TODO: this fusion currently does not happen properly, this is related to the clone
  // behavior when fusing into scf.forall.
  // Once fixed we'll be able to fuse.
  // Fusion will save us one roundtrip to memory.
  // transform.structured.fuse_into_containing_op %fill_linalg into %forall_reduction
  transform.structured.tile_to_forall_op %parallel_linalg num_threads [1, 4, 32]
      ( mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>] )
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)


  // Inability to tile reductions to scf.forall has 2 implications:
  //   1. since no scf.forall is present, no gpu.barrier is added.
  //      This should be fixed independently: ops that are not nested in an scf.forall
  //      should have a gpu.barrier. Later needs to be complemented by a barrier
  //      removal pass.
  //   2. Similarly, needs to be predicated under an if threadIx == 0 to avoid
  //      multiple threads updating the buffer inplace once bufferized.
  //
  // Instead, we can vectorize and go to vector SSA values that sidestep these
  // issues.
  // Everyone will race to the write while still computing the same value.
  //
  // That is still not good enough because we need to predicate this in order
  // to enable the parallel reduction on warps.

  // Step 3. Rank-reduce and vectorize.
  // ==================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
  transform.structured.vectorize %func : (!pdl.operation) -> !pdl.operation

  // Step 4. Bufferize and drop HAL decriptor from memref ops.
  // =========================================================
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> !pdl.operation
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op_2 : (!pdl.operation) -> !pdl.operation
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> !pdl.operation

  // Step 5. Post-bufferization mapping to blocks and threads.
  // =========================================================
  %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %func_3 = transform.iree.forall_to_workgroup %func_2 : (!pdl.operation) -> !pdl.operation
  transform.iree.map_nested_forall_to_gpu_threads %func_3
    { workgroup_dims = [32, 4, 1] }

  // Step 6. Post-bufferization vector distribution with rank-reduction.
  // ===================================================================
  %end_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %end_func { rank_reducing_linalg, rank_reducing_vector, fold_memref_aliases } : (!pdl.operation) -> ()
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
    : (!transform.any_op) -> !transform.any_op
  transform.iree.vector.warp_distribute %end_func
}
