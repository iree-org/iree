// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  // Step 1. Tile to forall with tile_sizes [2].
  // ===================================================
  %forall, %tiled_generic =
    transform.structured.tile_to_forall_op %matmul tile_sizes [2]
      ( mapping = [#gpu.block<x>] )
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
    : (!pdl.operation) -> ()

  // Step 2. Bufferize and drop HAL decriptor from memref ops.
  // =========================================================
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize %variant_op : (!pdl.operation) -> !pdl.operation
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

  // Step 3. Post-bufferization mapping workgroup.
  // =========================================================
  transform.iree.forall_to_workgroup %memref_func : (!pdl.operation) -> ()
}
