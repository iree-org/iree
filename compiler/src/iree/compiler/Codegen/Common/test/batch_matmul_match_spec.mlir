// RUN: iree-opt %s

module attributes { transform.with_named_sequence } {
transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks
    %0:2 = transform.iree.match_callback failures(propagate) "batch_matmul"(%root) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.emit_remark "fill" at %0#0 : !transform.any_op
    transform.iree.emit_remark "batch matmul" at %0#1 : !transform.any_op
    transform.yield
  } // @__transform_main
} // module
