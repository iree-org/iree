// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):

  // Step 1. Find the fill and matmul ops
  // ===========================================================================
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  // Step 2. Tile the matmul and fuse the fill
  // ===========================================================================
  %forall_grid, %grid_reduction =
  transform.structured.tile_to_forall_op %matmul tile_sizes [16] ( mapping = [#gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

  transform.structured.fuse_into_containing_op %fill into %forall_grid : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Promote operands in order to test loading from shared memory.
  %matmul_2 = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %promoted_matmul, %alloc_0, %alloc_1 =
    transform.iree.promote_operands %matmul_2 [0, 1] 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)


  // Step 3. Vectorize
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!transform.any_op) -> ()
  %func_3 = transform.structured.vectorize %func : (!transform.any_op) -> !transform.any_op

  // Step 4. Bufferize
  // ===========================================================================
  transform.iree.apply_patterns %func_3
    { fold_reassociative_reshapes, canonicalization, tiling_canonicalization, cse } : (!transform.any_op) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func_3 { erase_unnecessary_tensor_operands } : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()

  // Step 5. Pre-process the contract and transfer ops to put it in the right form.
  // ===========================================================================
  %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %func_2 {  prepare_vector_to_mma } : (!transform.any_op) -> ()

  // Step 6. Post-bufferization vector distribution
  // ===========================================================================
  %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_7
      workgroup_dims = [4, 8, 1] : (!transform.any_op) -> ()

  // Step 7. Do layout analysis and lower to mma
  // ===========================================================================
  %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  %func_11 = transform.iree.layout_analysis_and_distribution %func_10 : (!transform.any_op) -> (!transform.any_op)
}
