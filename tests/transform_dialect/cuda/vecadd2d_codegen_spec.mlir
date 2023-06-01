transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  // Step 1. Find three linalg.generics and tile to GPU thread blocks.
  // ===========================================================================
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %forall_grid, %_ = transform.structured.tile_to_forall_op %generics 
                  tile_sizes [5, 3] ( mapping = [#gpu.block<z>, #gpu.block<x>])
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()


  // Step 2. Rank reduce and bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!transform.any_op) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> !transform.any_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()

  // Step 3. Map to GPU thread blocks.
  // ===========================================================================
  transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
}
