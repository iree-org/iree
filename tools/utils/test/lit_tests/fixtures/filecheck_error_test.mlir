// RUN: iree-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @case1
func.func @case1() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case2
func.func @case2() {
  // CHECK: this pattern will not be found
  %c0 = arith.constant 0 : i32
  return
}

// -----

// CHECK-LABEL: @case3
func.func @case3() {
  // CHECK: return
  return
}
