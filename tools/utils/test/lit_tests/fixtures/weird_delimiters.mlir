// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @first
func.func @first() {
  return
}

//-----

// CHECK-LABEL: @second
func.func @second() {
  return
}

//   -----

// CHECK-LABEL: @third
func.func @third() {
  return
}
