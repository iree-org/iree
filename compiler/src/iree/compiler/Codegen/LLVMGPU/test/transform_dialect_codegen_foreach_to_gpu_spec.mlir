transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.fill"]} in %variant_op
  %foreach_thread, %tiled_fill = transform.structured.tile_to_foreach_thread_op %0 num_threads [5, 1] 
  ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )

  %1 = transform.structured.match ops{["linalg.matmul"]} in %variant_op
  %foreach_thread_2, %tiled_matmul = transform.structured.tile_to_foreach_thread_op %1 num_threads [7, 9]
  ( mapping = [#gpu.thread<x>, #gpu.thread<y>] )

  %variant_op_2 = transform.iree.bufferize %variant_op

  // Get the function to which to apply to.
  %2 = transform.structured.match ops{["linalg.matmul"]} in %variant_op_2
  %func = transform.get_closest_isolated_parent %2 : (!pdl.operation) -> !pdl.operation
  transform.iree.map_nested_foreach_thread_to_gpu_threads %func { workgroup_size = [10, 11]}
}
