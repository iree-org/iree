// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  %foreach_thread, %tiled_generic = transform.structured.tile_to_foreach_thread_op %0 num_threads [2]
  transform.iree.foreach_thread_to_flow %foreach_thread
}
