// RUN: iree-opt %s

module @user_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.print {name = "Hello Tuning Spec", skip_regions}
    transform.yield %arg0 : !transform.any_op
  }
}
