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
  %mfunc_0 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  %mfunc_1 = transform.apply_registered_pass "buffer-deallocation" to %mfunc_0 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %mfunc_1 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  %mfunc_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  %mfunc_3 = transform.apply_registered_pass "buffer-deallocation-simplification" to %mfunc_2 : (!transform.any_op) -> !transform.any_op
  %mfunc_4 = transform.apply_registered_pass "bufferization-lower-deallocations" to %mfunc_3 : (!transform.any_op) -> !transform.any_op
  transform.apply_cse to %mfunc_4 : !transform.any_op
  %mfunc_5 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %mfunc_5 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op

  // Step 3. Post-bufferization mapping workgroup.
  // =========================================================
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
}
