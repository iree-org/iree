// RUN: iree-opt %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks

    %fill, %convolution, %trailing =
      transform.iree.match_callback failures(propagate) "convolution"(%root)
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.iree.emit_remark "fill" at %fill : !transform.any_op
    transform.iree.emit_remark "convolution" at %convolution : !transform.any_op
    transform.iree.emit_remark "trailing" at %trailing : !transform.any_op
    transform.yield
  } // @__transform_main
} // module
