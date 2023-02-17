// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  // Step 1. Tile to forall with tile_sizes [2].
  // ===================================================
  %forall, %tiled_generic =
    transform.iree.tile_to_forall_and_workgroup_count_region %matmul tile_sizes [2]
      // TODO: IREE needs own workgroup mapping attribute.
      ( mapping = [#gpu.block<x>] )

  // Step 2. Bufferize and drop HAL decriptor from memref ops.
  // =========================================================
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %variant_op_3 = transform.iree.bufferize %variant_op_2
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func

  // Step 3. Post-bufferization mapping workgroup.
  // =========================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.forall_to_workgroup %func
}
