// RUN: iree-opt --iree-codegen-polynomial-approximation %s | FileCheck %s

// CHECK-LABEL: @polynomial_tan
func.func @polynomial_tan(%arg0: f32) -> f32 {
  // CHECK-NOT: math.tan
  %0 = math.tan %arg0 : f32 
  return %0 : f32
}
