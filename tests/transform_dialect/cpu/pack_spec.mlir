transform.sequence failures(propagate) {
^bb0(%variant_op: !pdl.operation):

  // Get pack op
  // =========================================================
  %pack = transform.structured.match ops{["tensor.pack"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  // Tile and distribute to workgroups
  // =========================================================
  %forall, %tiled_generic =
    transform.iree.tile_to_forall_and_workgroup_count_region %pack tile_sizes [2]
      // TODO: IREE needs own workgroup mapping attribute.
      ( mapping = [#gpu.block<x>] )

  // Bufferize and drop HAL decriptor from memref ops.
  // =========================================================
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize %variant_op : (!pdl.operation) -> !pdl.operation
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

  // Post-bufferization mapping workgroup.
  // =========================================================
  transform.iree.forall_to_workgroup %memref_func : (!pdl.operation) -> ()
}
