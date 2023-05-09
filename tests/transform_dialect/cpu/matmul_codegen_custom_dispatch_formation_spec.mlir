// RUN: iree-opt %s

transform.sequence failures(propagate) {  
^bb1(%variant_op: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  %forall, %tiled_generic =
    transform.structured.tile_to_forall_op %0 num_threads [2] 
    // TODO: IREE needs own workgroup mapping attribute.
    ( mapping = [#gpu.block<x>] )
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
      : (!pdl.operation) -> ()

  // Canonicalization/CSE is needed before bufferization otherwise unnecessary
  // allocs will be created.
  transform.iree.apply_patterns %variant_op 
    { canonicalization, tiling_canonicalization, cse } : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize %variant_op : (!pdl.operation) -> (!pdl.operation)
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()
  transform.iree.forall_to_workgroup %memref_func : (!pdl.operation) -> ()

  // CSE is needed on the workgroup_count region to pass this particular test.
  transform.iree.apply_patterns %variant_op_3 { cse } : (!pdl.operation) -> ()
}
