// RUN: iree-opt %s

module @user_spec attributes { transform.with_named_sequence } {
  transform.named_sequence @do_nothing(%op: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield %op : !transform.any_op
  }
  transform.named_sequence @lowering_strategy(%op: !transform.any_op {transform.readonly}) {
    transform.print {name = "I am external", skip_regions}
    transform.yield
  }
}
