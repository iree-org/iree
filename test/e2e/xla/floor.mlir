// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<4xf32> {
  %input = iree.unfoldable_constant dense<[0.0, 1.1, 2.5, 4.9]> : tensor<4xf32>
  %result = "xla_hlo.floor"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: 4xf32=0 1 2 4

// -----

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<f32> {
  %input = iree.unfoldable_constant dense<101.3> : tensor<f32>
  %result = "xla_hlo.floor"(%input) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=101

// -----

// CHECK-LABEL: EXEC @double
func @double() -> tensor<f64> {
  %input = iree.unfoldable_constant dense<11.2> : tensor<f64>
  %result = "xla_hlo.floor"(%input) : (tensor<f64>) -> tensor<f64>
  return %result : tensor<f64>
}
// CHECK: f32=11

// -----

// CHECK-LABEL: EXEC @negative
func @negative() -> tensor<f32> {
  %input = iree.unfoldable_constant dense<-1.1> : tensor<f32>
  %result = "xla_hlo.floor"(%input) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=-2
