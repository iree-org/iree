// RUN: iree-opt %s

module @mmt_tile_and_fuse_spec attributes { transform.with_named_sequence } {
  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly}) -> ()
    attributes { iree_codegen.tuning_spec_entrypoint } {
    %mmt = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // transform.print %mmt {name="MMT"} : !transform.any_op
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0],
                                                   reduction = [0, 0, 4],
                                                   thread = [8, 4],
                                                   promote_operands = [0, 1]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [128, 1, 1] subgroup_size = 64>
    > -> !transform.any_param
    transform.annotate %mmt "compilation_info" = %config : !transform.any_op, !transform.any_param
    // Add a dummy unit attribute to be sure that the tuning spec applied.
    // Otherwise it would be difficult to tell if the lowering config attribute
    // comes from our tuning spec or if the compiler heuristic happened to produce
    // the same config as this script.
    transform.annotate %mmt "__tuning_spec_applied__" : !transform.any_op
    transform.yield
  }
}
