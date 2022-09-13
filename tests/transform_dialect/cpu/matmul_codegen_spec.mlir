// RUN: iree-opt %s

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 failures(propagate) {
  ^bb1(%variant_op: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %variant_op

    %foreach_thread, %tiled_generic =
      transform.structured.tile_to_foreach_thread_op %0 tile_sizes [4]

    transform.print {name = "after for_each"}

    %1 = transform.iree.bufferize %variant_op

    transform.print {name = "after bufferize"}

    %func = transform.structured.match ops{["func.func"]} in %1
    transform.iree.foreach_thread_to_workgroup %func
  }
}
