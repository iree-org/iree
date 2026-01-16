// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-codegen-remove-index-hints))' %s | FileCheck %s

// Test: index_hint with lane_constant is removed.
// CHECK-LABEL: func.func @remove_lane_constant_hint
// CHECK-NOT: iree_codegen.index_hint
// CHECK: return %arg0
func.func @remove_lane_constant_hint(%arg0: index) -> index {
  %hint = iree_codegen.index_hint %arg0(#iree_gpu.lane_constant<16>) : index
  return %hint : index
}

// -----

// Test: index_hint with lane_increment is removed.
// CHECK-LABEL: func.func @remove_lane_increment_hint
// CHECK-NOT: iree_codegen.index_hint
// CHECK: return %arg0
func.func @remove_lane_increment_hint(%arg0: index) -> index {
  %hint = iree_codegen.index_hint %arg0(#iree_gpu.lane_increment<16>) : index
  return %hint : index
}

// -----

// Test: Multiple hints in sequence are all removed.
// CHECK-LABEL: func.func @remove_multiple_hints
// CHECK-NOT: iree_codegen.index_hint
// CHECK: arith.addi %arg0, %arg1
func.func @remove_multiple_hints(%arg0: index, %arg1: index) -> index {
  %hint0 = iree_codegen.index_hint %arg0(#iree_gpu.lane_constant<16>) : index
  %hint1 = iree_codegen.index_hint %arg1(#iree_gpu.lane_increment<16>) : index
  %sum = arith.addi %hint0, %hint1 : index
  return %sum : index
}
