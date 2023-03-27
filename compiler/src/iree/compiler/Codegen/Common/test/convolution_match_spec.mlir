// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.iree.register_match_callbacks

  %fill, %convolution, %trailing =
    transform.iree.match_callback failures(propagate) "convolution"(%arg0)
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)

  transform.iree.emit_remark "fill" at %fill : !pdl.operation
  transform.iree.emit_remark "convolution" at %convolution : !pdl.operation
  transform.iree.emit_remark "trailing" at %trailing : !pdl.operation
}
