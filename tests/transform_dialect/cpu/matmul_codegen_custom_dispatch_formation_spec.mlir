// RUN: iree-opt %s

transform.sequence failures(propagate) {  
^bb1(%variant_op: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  %forall, %tiled_generic =
    transform.structured.tile_to_forall_op %0 num_threads [2] 
    // TODO: IREE needs own workgroup mapping attribute.
    ( mapping = [#gpu.block<x>] )

  // Canonicalization/CSE is needed before bufferization otherwise unnecessary
  // allocs will be created.
  %variant_op_2 = transform.iree.apply_patterns %variant_op 
    { canonicalization, tiling_canonicalization, cse }
  %variant_op_3 = transform.iree.bufferize %variant_op_2
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  %memref_func_2 = transform.iree.erase_hal_descriptor_type_from_memref %memref_func
  %memref_func_3 = transform.iree.forall_to_workgroup %memref_func_2

  // CSE is needed on the workgroup_count region to pass this particular test.
  transform.iree.apply_patterns %variant_op_3 { cse }
}
