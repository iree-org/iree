// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --target_backends=vulkan-spirv %s --output_types="i" | IreeFileCheck %s)

// TODO(b/146030213) : This test fails cause the initialization isn't
// done correctly within the vulkan backend. Enable this test once that
// is done.

// Int sum values from [1, 10]
// CHECK-LABEL: EXEC @reduce_sum_1x10xi32
func @reduce_sum_1x10xi32() -> tensor<1xi32> {
  %0 = constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = constant dense<30> : tensor<i32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.max"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  return %2 : tensor<1xi32>
}
// TO-CHECK: 1xi32=30
