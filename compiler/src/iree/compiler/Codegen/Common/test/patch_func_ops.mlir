// RUN: iree-opt %s --iree-codegen-debug-patch-func-ops --iree-codegen-debug-patched-func-ops-file-name=%p/patch_func_ops_spec.mlir | FileCheck %s

func.func @double_index(%arg0: index) -> index {
  %c2 = arith.constant 2 : index
  %0 = arith.muli %arg0, %c2 : index
  return %0 : index
}
// CHECK-LABEL: @double_index
// CHECK:         arith.addi

func.func @no_matching_func_op(%arg0: index) -> index {
  return %arg0 : index
}
// CHECK-LABEL: @no_matching_func_op
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:       return %[[ARG0]]

func.func @index_times_four(%arg0: index) -> index {
  %c4 = arith.constant 4 : index
  %0 = arith.muli %arg0, %c4 : index
  return %0 : index
}
// CHECK-LABEL: @index_times_four
// CHECK:         arith.addi
// CHECK:         arith.addi
