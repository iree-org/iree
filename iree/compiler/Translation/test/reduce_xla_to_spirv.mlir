// RUN: iree-opt -iree-hlo-reduction-to-linalg -iree-xla-to-linalg-to-spirv %s
// TODO(hanchung): Remove the test once the passes has been integrated within
// execution runtime.

// CHECK: spv.module
module {
  // CHECK: spv.loop
  func @reduction_entry(memref<5x4xf32>, memref<f32>, memref<4xf32>)
  attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 1 : i32, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[4, 5, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32}

  func @reduction_apply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = xla_hlo.min %arg0, %arg1 : tensor<f32>
    %1 = xla_hlo.max %arg0, %arg1 : tensor<f32>
    %2 = xla_hlo.add %0, %1 : tensor<f32>
    return %2 : tensor<f32>
  }
}
