// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

func.func @lower_value_barrier(%input: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_gpu.value_barrier %input : vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_value_barrier
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_value_barrier
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: vector<4xf32>
//  CHECK-NEXT:   gpu.barrier
//  CHECK-NEXT:   return %[[INPUT]]
