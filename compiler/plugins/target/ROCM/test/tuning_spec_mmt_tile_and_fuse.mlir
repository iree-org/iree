// RUN: iree-opt %s

module @mmt_tile_and_fuse_spec attributes { transform.with_named_sequence } {
transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                          %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "__custom_tuning_spec_applied__" : !transform.any_op
  transform.yield
}

transform.named_sequence @match_mmt(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  transform.match.operation_name %matmul ["linalg.generic"] : !transform.any_op
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0],
                                                  reduction = [0, 0, 4],
                                                  thread = [8, 4],
                                                  promote_operands = [0, 1]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
      workgroup_size = [128, 1, 1] subgroup_size = 64>
  > -> !transform.any_param
 transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @main(%variant_op: !transform.any_op {transform.consumed}) -> (!transform.any_op)
  attributes { iree_codegen.tuning_spec_entrypoint } {
  transform.print %variant_op {name="Custom spec"} : !transform.any_op
  %res = transform.foreach_match in %variant_op
    @match_mmt -> @apply_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}
}
