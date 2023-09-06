// RUN: iree-opt --iree-codegen-polynomial-approximation --split-input-file %s | FileCheck %s

// CHECK-LABEL: @polynomial_tan
func.func @polynomial_tan(%arg0: f32) -> f32 {
  // CHECK-NOT: math.tan
  %0 = math.tan %arg0 : f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @expanded_pow
func.func @expanded_pow(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK-NOT: math.pow
  %0 = math.powf %arg0, %arg1 : f32
  return %0 : f32
}
