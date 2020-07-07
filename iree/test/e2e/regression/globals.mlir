// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

module {
  flow.variable @counter mutable dense<2.0> : tensor<f32>

  // CHECK: EXEC @get_state
  func @get_state() -> tensor<f32> {
    %0 = flow.variable.load @counter : tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK: f32=2

  // CHECK: EXEC @inc
  func @inc() -> tensor<f32> {
    %0 = flow.variable.load @counter : tensor<f32>
    %c1 = constant dense<1.0> : tensor<f32>
    %1 = mhlo.add %0, %c1 : tensor<f32>
    flow.variable.store %1, @counter : tensor<f32>
    %2 = flow.variable.load @counter : tensor<f32>
    return %2 : tensor<f32>
  }
  // CHECK: f32=3
}
