transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  // Step 1. Find three linalg.generics and tile to GPU thread blocks.
  // ===========================================================================
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %_, %forall_grid = transform.structured.tile_using_forall %generics 
                  tile_sizes [5, 3] ( mapping = [#gpu.block<z>, #gpu.block<x>])
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()


  // Step 2. Rank reduce and bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    transform.apply_patterns.vector.cast_away_vector_leading_one_dim
  } : !transform.any_op
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> !transform.any_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op

  // Step 3. Map to GPU thread blocks.
  // ===========================================================================
  transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
}
