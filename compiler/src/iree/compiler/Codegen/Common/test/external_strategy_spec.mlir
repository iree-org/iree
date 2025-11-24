// RUN: iree-opt %s | FileCheck %s

// This file defines an external strategy spec used by lowering_config_interpreter.mlir
// to test importing external lowering strategies via --iree-codegen-tuning-spec-path.

// CHECK: module @user_spec
// CHECK-SAME: transform.with_named_sequence
module @user_spec attributes { transform.with_named_sequence } {
  // CHECK: transform.named_sequence @lowering_strategy
  transform.named_sequence @lowering_strategy(%op: !transform.any_op {transform.readonly}) {
    // CHECK: transform.print {name = "I am external", skip_regions}
    transform.print {name = "I am external", skip_regions}
    transform.yield
  }
  // CHECK: transform.named_sequence @import_lowering_strategy
  // CHECK-SAME: iree_codegen.tuning_spec_entrypoint
  transform.named_sequence @import_lowering_strategy(%op: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    // CHECK: transform.util.create_serialized_module
    %syms = transform.util.create_serialized_module {
      ^bb0(%m: !transform.any_op):
        // CHECK: transform.util.import_symbol @lowering_strategy
        transform.util.import_symbol @lowering_strategy into %m if undefined : (!transform.any_op) -> !transform.any_op
        // CHECK: transform.annotate {{.+}} "transform.with_named_sequence"
        transform.annotate %m "transform.with_named_sequence" : !transform.any_op
    } -> !transform.any_param
    // CHECK: transform.annotate {{.+}} "iree_codegen_external_symbols"
    transform.annotate %op "iree_codegen_external_symbols" = %syms : !transform.any_op, !transform.any_param
    transform.yield %op : !transform.any_op
  }
}
