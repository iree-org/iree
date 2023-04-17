// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.iree.register_match_callbacks

  %fill, %convolution, %trailing =
    transform.iree.match_callback failures(propagate) "convolution"(%arg0)
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

  transform.iree.emit_remark "fill" at %fill : !transform.any_op
  transform.iree.emit_remark "convolution" at %convolution : !transform.any_op
  transform.iree.emit_remark "trailing" at %trailing : !transform.any_op
}
