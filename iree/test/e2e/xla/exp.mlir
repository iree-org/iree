// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: iree-run-mlir -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv -iree-use-linalg-to-spirv-path %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<4xf32> {
  %input = iree.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf32>
  %result = "xla_hlo.exp"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: 4xf32=1 2.71828 7.38906 54.5981

// -----

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<f32> {
  %input = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %result = "xla_hlo.exp"(%input) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=2.71828

// -----

// CHECK-LABEL: EXEC @double
func @double() -> tensor<f64> {
  %input = iree.unfoldable_constant dense<1.0> : tensor<f64>
  %result = "xla_hlo.exp"(%input) : (tensor<f64>) -> tensor<f64>
  return %result : tensor<f64>
}
// CHECK: f32=2.71828

// -----

// CHECK-LABEL: EXEC @negative
func @negative() -> tensor<f32> {
  %input = iree.unfoldable_constant dense<-1.0> : tensor<f32>
  %result = "xla_hlo.exp"(%input) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=0.367879
