module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %foreach_op = transform.structured.tile_using_forall %0 num_threads [42, 67]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %dispatch_op = transform.iree.forall_to_flow %foreach_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
