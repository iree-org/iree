// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %variant_op

  %foreach_thread, %tiled_generic =
    transform.structured.tile_to_foreach_thread_op %0 num_threads [2] 
    ( mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>] )

  %1 = transform.iree.bufferize %variant_op

  %func = transform.structured.match ops{["func.func"]} in %1
  transform.iree.foreach_thread_to_workgroup %func
}
