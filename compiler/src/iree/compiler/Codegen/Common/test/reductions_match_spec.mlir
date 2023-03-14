// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.iree.register_match_callbacks

  %leading, %fill, %reduction, %trailing =
    transform.iree.match_callback failures(propagate) "reduction"(%arg0)
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

  transform.iree.emit_remark "leading" at %leading : !pdl.operation
  transform.iree.emit_remark "fill" at %fill : !pdl.operation
  transform.iree.emit_remark "reduction" at %reduction : !pdl.operation
  transform.iree.emit_remark "trailing" at %trailing : !pdl.operation
}
