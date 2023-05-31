// RUN: iree-run-mlir --Xcompiler,iree-input-type=mhlo_legacy --Xcompiler,iree-hal-target-backends=vmvx %s | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-input-type=mhlo_legacy --Xcompiler,iree-hal-target-backends=vulkan-spirv %s | FileCheck %s)

module {
  ml_program.global private mutable @counter(dense<2.0> : tensor<f32>): tensor<f32>

  // CHECK: EXEC @get_state
  func.func @get_state() -> tensor<f32> {
    %0 = ml_program.global_load @counter : tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK: f32=2

  // CHECK: EXEC @inc
  func.func @inc() -> tensor<f32> {
    %0 = ml_program.global_load @counter : tensor<f32>
    %c1 = arith.constant dense<1.0> : tensor<f32>
    %1 = mhlo.add %0, %c1 : tensor<f32>
    ml_program.global_store @counter = %1 : tensor<f32>
    %2 = ml_program.global_load @counter : tensor<f32>
    return %2 : tensor<f32>
  }
  // CHECK: f32=3
}
