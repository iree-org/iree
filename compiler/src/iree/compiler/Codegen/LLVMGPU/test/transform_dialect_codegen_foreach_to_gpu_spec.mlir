transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %forall, %tiled_fill = transform.structured.tile_to_forall_op %0 num_threads [5, 1] 
  ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )

  %1 = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %forall_2, %tiled_matmul = transform.structured.tile_to_forall_op %1 num_threads [7, 9]
  ( mapping = [#gpu.thread<x>, #gpu.thread<y>] )

  // Canonicalization/CSE is needed before bufferization otherwise unnecessary
  // allocs will be created.
  %func = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func 
    { fold_reassociative_reshapes, canonicalization, tiling_canonicalization, cse }
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %variant_op_3 = transform.iree.bufferize %variant_op_2
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func

  // Get the function to which to apply to.
  %func_2 = transform.structured.match ops{["linalg.matmul"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %func_3 = transform.get_closest_isolated_parent %func_2 : (!pdl.operation) -> !pdl.operation
  %func_4 = transform.iree.map_nested_forall_to_gpu_threads %func_3 { workgroup_size = [10, 11]}

  // Late canonicalizations to cleanup and pass the checks
  %func_5 = transform.iree.apply_patterns %func_4
    { canonicalization, tiling_canonicalization, licm, cse }
}
