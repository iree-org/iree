transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %foreach_op, %tiled_op = transform.structured.tile_to_forall_op %0 num_threads [42, 67]
  %dispatch_op = transform.iree.forall_to_flow %foreach_op
}
