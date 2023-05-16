// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=vmvx %s | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-hal-target-backends=vulkan-spirv %s | FileCheck %s)

util.global private mutable @counter = dense<2.0> : tensor<f32>

// CHECK: EXEC @inc
func.func @inc() -> tensor<f32> {
  %0 = util.global.load @counter : tensor<f32>
  %c1 = arith.constant dense<1.0> : tensor<f32>
  %1 = arith.addf %0, %c1 : tensor<f32>
  util.global.store %1, @counter : tensor<f32>
  %2 = util.global.load @counter : tensor<f32>
  return %2 : tensor<f32>
}
// CHECK: f32=3
