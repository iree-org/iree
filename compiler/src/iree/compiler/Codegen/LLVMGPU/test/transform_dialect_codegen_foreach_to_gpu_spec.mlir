transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %forall, %tiled_fill = transform.structured.tile_to_forall_op %0 num_threads [5, 1] 
  ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %1 = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %forall_2, %tiled_matmul = transform.structured.tile_to_forall_op %1 num_threads [7, 9]
  ( mapping = [#gpu.thread<x>, #gpu.thread<y>] )
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Canonicalization/CSE is needed before bufferization otherwise unnecessary
  // allocs will be created.
  %func = transform.structured.match ops{["func.func"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %func 
    { fold_reassociative_reshapes, canonicalization, tiling_canonicalization, cse } : (!transform.any_op) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> (!transform.any_op)
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %memref_func 
    workgroup_dims = [10, 11, 1] : (!transform.any_op) -> ()

  // Late canonicalizations to cleanup and pass the checks
  transform.iree.apply_patterns %memref_func
    { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()
}
