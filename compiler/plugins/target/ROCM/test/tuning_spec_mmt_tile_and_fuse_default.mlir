// RUN: iree-opt %s

module @mmt_tile_and_fuse_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint} {
transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                          %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "__custom_tuning_spec_applied__" : !transform.any_op
  transform.yield
}

transform.named_sequence @match_mmt(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  transform.match.operation_name %matmul ["linalg.generic"] : !transform.any_op
  %config = transform.param.constant {key = "custom_config"} -> !transform.any_param
 transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> (!transform.any_op)
  attributes { iree_codegen.tuning_spec_entrypoint } {
  %res = transform.foreach_match in %variant_op
    @match_mmt -> @apply_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}
}
