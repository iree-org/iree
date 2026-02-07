module @matmul_spec attributes { transform.with_named_sequence } {
transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                          %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "__custom_tuning_spec_applied__" : !transform.any_op
  transform.yield
}

transform.named_sequence @match_matmul(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_cpu.lowering_config<
      distribution = [64, 64, 0],
      vector_common_parallel = [8, 16, 0],
      vector_reduction = [0, 0, 8]>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
  > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @main(%variant_op: !transform.any_op {transform.consumed}) -> (!transform.any_op)
  attributes { iree_codegen.tuning_spec_entrypoint } {
  %res = transform.foreach_match in %variant_op
    @match_matmul -> @apply_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}
}
