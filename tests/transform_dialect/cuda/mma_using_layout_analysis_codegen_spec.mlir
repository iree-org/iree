// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):

  // Step 1. Find the fill and matmul ops
  // ===========================================================================
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  // Step 2. Tile the matmul and fuse the fill
  // ===========================================================================
  %forall_grid, %grid_reduction =
  transform.iree.tile_to_forall_and_workgroup_count_region %matmul tile_sizes [16] ( mapping = [#gpu.block<x>] )
  transform.structured.fuse_into_containing_op %fill into %forall_grid

  // Step 3. Vectorize
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %func_2 = transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector }
  %func_3 = transform.structured.vectorize %func_2

  // Step 4. Bufferize
  // ===========================================================================
   %func_4 = transform.iree.apply_patterns %func_3
    { fold_reassociative_reshapes, canonicalization, tiling_canonicalization, cse }
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %func_5 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!pdl.operation) -> !pdl.operation
  %func_6 = transform.iree.apply_patterns %func_5 { erase_unnecessary_tensor_operands }
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op_2
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func

  // Step 6. Post-bufferization vector distribution
  // ===========================================================================
  %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %func_8 = transform.iree.forall_to_workgroup %func_7
  %func_9 = transform.iree.map_nested_forall_to_gpu_threads %func_8
      { workgroup_size = [4, 8, 1] }

  // Step 7. Do layout analysis and lower to mma
  // ===========================================================================
  %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %func_11 = transform.iree.layout_analysis_and_distribution %func_10 : (!pdl.operation) -> (!pdl.operation)
}
