// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op

  %foreach_thread, %tiled_generic =
    transform.iree.tile_to_foreach_thread_and_workgroup_count_region %matmul tile_sizes [2]
      // TODO: IREE needs own workgroup mapping attribute.
      ( mapping = [#gpu.block<x>] )

  %variant_op_2 = transform.iree.bufferize %variant_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_2
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func

  %func = transform.structured.match ops{["func.func"]} in %variant_op_2
  transform.iree.foreach_thread_to_workgroup %func
}
