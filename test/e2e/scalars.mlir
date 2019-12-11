// RUN: iree-run-mlir %s | IreeFileCheck %s

// CHECK-LABEL: EXEC @scalars
func @scalars() -> tensor<f32> {
  %0 = constant dense<2.0> : tensor<f32>
  return %0 : tensor<f32>
}
// CHECK: f32=2
