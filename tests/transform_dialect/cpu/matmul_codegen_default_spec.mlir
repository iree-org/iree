// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  // Step 1. Tile to forall with tile_sizes [2].
  // ===================================================
  %forall, %tiled_generic =
    transform.structured.tile_to_forall_op %matmul tile_sizes [2]
      ( mapping = [#gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
    : (!transform.any_op) -> ()

  // Step 2. Bufferize and drop HAL decriptor from memref ops.
  // =========================================================
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()

  // Step 3. Post-bufferization mapping workgroup.
  // =========================================================
  transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
}
