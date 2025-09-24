// RUN: iree-opt %s

module @user_spec attributes { transform.with_named_sequence } {
  transform.named_sequence @lowering_strategy(%op: !transform.any_op {transform.readonly}) {
    transform.print {name = "I am external", skip_regions}
    transform.yield
  }
  transform.named_sequence @import_lowering_strategy(%op: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    %syms = transform.util.create_serialized_module {
      ^bb0(%m: !transform.any_op):
        transform.util.import_symbol @lowering_strategy into %m if undefined : (!transform.any_op) -> !transform.any_op
        transform.annotate %m "transform.with_named_sequence" : !transform.any_op
    } -> !transform.any_param
    transform.annotate %op "iree_codegen_external_symbols" = %syms : !transform.any_op, !transform.any_param
    transform.yield %op : !transform.any_op
  }
}
