// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @case_one
// This case has an in-body RUN directive used by lit
// RUN: echo "alpha" | FileCheck %s --check-prefix=ALPHA
// ALPHA: alpha
func.func @case_one() {
  // CHECK: return
  return
}

// -----

// CHECK-LABEL: @case_two
// Another in-body RUN directive
// RUN: echo "beta" | FileCheck %s --check-prefix=BETA
// BETA: beta
func.func @case_two() {
  // CHECK: return
  return
}
