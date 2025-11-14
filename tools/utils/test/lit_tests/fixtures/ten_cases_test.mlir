// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @case_one
func.func @case_one() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_two
func.func @case_two() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_three
func.func @case_three() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_four
func.func @case_four() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_five
func.func @case_five() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_six
func.func @case_six() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_seven
func.func @case_seven() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_eight
func.func @case_eight() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_nine
func.func @case_nine() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_ten
func.func @case_ten() {
  // CHECK: return
  return
}
