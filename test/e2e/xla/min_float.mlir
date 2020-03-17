// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<4xf32> {
  %lhs = iree.unfoldable_constant dense<[1.0, 2.0, 7.0, 4.0]> : tensor<4xf32>
  %rhs = iree.unfoldable_constant dense<[5.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "xla_hlo.minimum"(%lhs, %rhs) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: 4xf32=1 2 3 4

// -----

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<f32> {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %rhs = iree.unfoldable_constant dense<2.0> : tensor<f32>
  %result = "xla_hlo.minimum"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=1

// -----

// CHECK-LABEL: EXEC @double
func @double() -> tensor<f64> {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f64>
  %rhs = iree.unfoldable_constant dense<2.0> : tensor<f64>
  %result = "xla_hlo.minimum"(%lhs, %rhs) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  return %result : tensor<f64>
}
// CHECK: f32=1

// -----

// CHECK-LABEL: EXEC @negative
func @negative() -> tensor<f32> {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %rhs = iree.unfoldable_constant dense<-2.0> : tensor<f32>
  %result = "xla_hlo.minimum"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=-2
