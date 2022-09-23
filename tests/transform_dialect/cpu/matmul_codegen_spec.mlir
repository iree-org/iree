// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %variant_op
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op

  %foreach_thread, %tiled_generic =
    transform.iree.tile_to_workgroups_op %matmul %func tile_sizes [2]

  %variant_op_2 = transform.iree.bufferize %variant_op

  %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_2
  transform.iree.foreach_thread_to_workgroup %func_2
}
