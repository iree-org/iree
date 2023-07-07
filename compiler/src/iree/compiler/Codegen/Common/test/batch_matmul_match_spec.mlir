// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.iree.register_match_callbacks
  %0:2 = transform.iree.match_callback failures(propagate) "batch_matmul"(%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.emit_remark "fill" at %0#0 : !transform.any_op
  transform.iree.emit_remark "batch matmul" at %0#1 : !transform.any_op
}
