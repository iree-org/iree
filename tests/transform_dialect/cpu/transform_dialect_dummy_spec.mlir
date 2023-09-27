// RUN: iree-opt %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @print_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.print %variant_op {name = "from_config"} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @print_selected4(%variant_op: !transform.any_op {transform.consumed}) {
    transform.print %variant_op {name = "from_selected4"} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @print_selected6(%variant_op: !transform.any_op {transform.consumed}) {
    transform.print %variant_op {name = "from_selected6"} : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    print %arg0 {name = "from_flag"} : !transform.any_op
    transform.yield
  }
}
