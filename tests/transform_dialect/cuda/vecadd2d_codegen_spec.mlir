transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%varop: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %varop
  %foreach, %tiled = transform.iree.tile_to_foreach_thread_and_workgroup_count_region %generic tile_sizes [16, 4] ( mapping = [#gpu.block<z>, #gpu.block<x>] )
  %funcx = transform.structured.match ops{["func.func"]} in %varop
  transform.iree.apply_patterns %funcx { rank_reducing }
  %variant_op_2 = transform.iree.bufferize { target_gpu } %varop
  %func_4 = transform.structured.match ops{["func.func"]} in %variant_op_2
  %func_5 = transform.iree.foreach_thread_to_workgroup %func_4  
}
