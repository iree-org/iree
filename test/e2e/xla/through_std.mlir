// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @xla_through_stdops
func @xla_through_stdops () -> (tensor<f32>, tensor<f32>) {
  %tf32 = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %0 = "xla_hlo.add"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "xla_hlo.mul"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0, %1 : tensor<f32>, tensor<f32>
}
// CHECK: f32=2
// CHECK-NEXT: f32=1
