// RUN: iree-opt %s

transform.sequence failures(propagate) {  
^bb1(%variant_op: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  %forall, %tiled_generic =
    transform.structured.tile_to_forall_op %0 num_threads [2] 
    // TODO: IREE needs own workgroup mapping attribute.
    ( mapping = [#gpu.block<x>] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
      : (!transform.any_op) -> ()

  // Canonicalization/CSE is needed before bufferization otherwise unnecessary
  // allocs will be created.
  transform.iree.apply_patterns %variant_op 
    { canonicalization, tiling_canonicalization, cse } : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> (!transform.any_op)
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()
  transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()

  // CSE is needed on the workgroup_count region to pass this particular test.
  transform.iree.apply_patterns %variant_op_3 { cse } : (!transform.any_op) -> ()
}
