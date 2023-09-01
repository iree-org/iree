// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.iree.register_match_callbacks

  %leading, %fill, %reduction, %trailing =
    transform.iree.match_callback failures(propagate) "reduction"(%arg0)
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  transform.iree.emit_remark "leading" at %leading : !transform.any_op
  transform.iree.emit_remark "fill" at %fill : !transform.any_op
  transform.iree.emit_remark "reduction" at %reduction : !transform.any_op
  transform.iree.emit_remark "trailing" at %trailing : !transform.any_op
}
