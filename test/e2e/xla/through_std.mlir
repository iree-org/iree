// RUN: iree-run-mlir %s | IreeFileCheck %s

// CHECK-LABEL: EXEC @xla_through_stdops
func @xla_through_stdops () -> (tensor<f32>, tensor<f32>) {
  %tf32 = constant dense<1.0> : tensor<f32>
  %0 = "xla_hlo.add"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "xla_hlo.mul"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0, %1 : tensor<f32>, tensor<f32>
}
// CHECK: f32=2
// CHECK-NEXT: f32=1
