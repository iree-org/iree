transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  // Step 1. Find three linalg.generics and tile to GPU thread blocks.
  // ===========================================================================
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %forall_grid, %_ = transform.structured.tile_to_forall_op %generics 
                  tile_sizes [5, 3] ( mapping = [#gpu.block<z>, #gpu.block<x>])
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!pdl.operation) -> ()


  // Step 2. Rank reduce and bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!pdl.operation) -> !pdl.operation
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

  // Step 3. Map to GPU thread blocks.
  // ===========================================================================
  transform.iree.forall_to_workgroup %memref_func : (!pdl.operation) -> ()
}
