// RUN: iree-opt %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @codegen(
      %variant_op: !transform.any_op {transform.consumed}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Step 1. Tile to forall with tile_sizes [2].
    // ===================================================
    %tiled_generic, %forall =
      transform.structured.tile_using_forall %matmul tile_sizes [2]
        ( mapping = [#gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
      : (!transform.any_op) -> ()

    // Step 2. Bufferize and drop HAL decriptor from memref ops.
    // =========================================================
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op

    // Step 3. Post-bufferization mapping workgroup.
    // =========================================================
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.yield
  }
} // module
