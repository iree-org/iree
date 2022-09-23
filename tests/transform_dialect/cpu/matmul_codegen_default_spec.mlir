// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %variant_op
  %0 = transform.structured.match ops{["linalg.matmul"]} in %variant_op

  %foreach_thread, %tiled_generic =
    transform.iree.tile_to_workgroups_op %0 %func tile_sizes [2]

  %1 = transform.iree.bufferize %variant_op
}
