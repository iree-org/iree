// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @foo
func.func @foo() {
  return
}

// -----
// CHECK-LABEL: @foo
func.func @foo() {
  // Second function with same label.
  return
}

// -----
// CHECK-LABEL: @bar
func.func @bar() {
  return
}
