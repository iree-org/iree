// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @case_one
func.func @case_one() {
  return
}

// -----
// CHECK-LABEL: @case_two
func.func @case_two() {
  return
}

// -----
// CHECK-LABEL: @case_three
func.func @case_three() {
  return
}

// -----
// CHECK-LABEL: @case_four
func.func @case_four() {
  return
}

// -----
// CHECK-LABEL: @case_five
func.func @case_five() {
  return
}
