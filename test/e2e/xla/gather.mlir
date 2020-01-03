// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s -input-value="5x1x5xi32=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25]" -input-value="i64=2" | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s -input-value="5x1x5xi32=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25]" -input-value="i64=2" | IreeFileCheck %s)

module {
  // CHECK-LABEL: EXEC @foo
  func @foo(%arg0: tensor<5x1x5xi32>, %arg1: tensor<i64>) -> tensor<1x5xi32> {
    %1 = "xla_hlo.gather"(%arg0, %arg1) {
      dimension_numbers = {
        collapsed_slice_dims = dense<0> : tensor<1xi64>,
        index_vector_dim = 0 : i64,
        offset_dims = dense<[0, 1]> : tensor<2xi64>,
        start_index_map = dense<0> : tensor<1xi64>
      },
      slice_sizes = dense<[1, 1, 5]> : tensor<3xi64>
    } : (tensor<5x1x5xi32>, tensor<i64>) -> tensor<1x5xi32>
    return %1 : tensor<1x5xi32>
  }
  // CHECK 1x5xi32=[11 12 13 14 15]
}
