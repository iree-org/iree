// RUN: iree-opt %s

module @user_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.print {name = "Hello Tuning Spec"}
    transform.yield %arg : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
    transform.yield
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
      %res = transform.foreach_match in %arg0
        @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
  }
}
