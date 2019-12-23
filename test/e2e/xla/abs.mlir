// RUN: iree-run-mlir2 -iree-hal-target-backends=interpreter-bytecode -input-value="2x2xf32= 1 2 3 4" -input-value="2x3xf32= 5 6 7 8 9 10" -input-value="2x2xf32= 11 12 13 14" %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir2 -iree-hal-target-backends=vulkan-spirv -input-value="2x2xf32= 1 2 3 4" -input-value="2x3xf32= 5 6 7 8 9 10" -input-value="2x2xf32= 11 12 13 14" %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<4xf32> {
  %input = constant dense<[-1.0, -2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "xla_hlo.abs"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: 4xf32=1 2 3 4

// -----

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<f32> {
  %input = constant dense<-4.0> : tensor<f32>
  %result = "xla_hlo.abs"(%input) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=4
