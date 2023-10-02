// RUN: iree-opt %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.print %root : !transform.any_op
    transform.yield 
  }
} // module
