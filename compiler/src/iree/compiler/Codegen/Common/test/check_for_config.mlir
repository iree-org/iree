// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file --verify-diagnostics | FileCheck %s

func.func @no_lowering_config() -> index {
  %0 = arith.constant 0 : index
  return %0 : index
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.constant"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.iree.match.has_no_lowering_config %0 : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: no_lowering_config

// -----

func.func @lowering_config() -> index {
  %0 = arith.constant {lowering_config = #iree_codegen.lowering_config<tile_sizes = []>} 0 : index
  return %0 : index
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.constant"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{payload has a lowering config or compilation info.}}
    transform.iree.match.has_no_lowering_config %0 : !transform.any_op
    transform.yield
  }
}

// -----

func.func @compilation_info() -> index {
  %0 = arith.constant {compilation_info = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = []>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>>} 0 : index
  return %0 : index
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.constant"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{payload has a lowering config or compilation info.}}
    transform.iree.match.has_no_lowering_config %0 : !transform.any_op
    transform.yield
  }
}
