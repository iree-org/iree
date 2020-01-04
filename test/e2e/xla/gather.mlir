// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

module {
  // CHECK-LABEL: EXEC @foo
  func @foo() -> tensor<1x5xi32> {
    %input = iree.unfoldable_constant dense<[
      [[01, 02, 03, 04, 05]],
      [[06, 07, 08, 09, 10]],
      [[11, 12, 13, 14, 15]],
      [[16, 17, 18, 19, 20]],
      [[21, 22, 23, 24, 25]]
    ]> : tensor<5x1x5xi32>
    %start_indices = iree.unfoldable_constant dense<2> : tensor<i64>
    %res = "xla_hlo.gather"(%input, %start_indices) {
      dimension_numbers = {
        collapsed_slice_dims = dense<0> : tensor<1xi64>,
        index_vector_dim = 0 : i64,
        offset_dims = dense<[0, 1]> : tensor<2xi64>,
        start_index_map = dense<0> : tensor<1xi64>
      },
      slice_sizes = dense<[1, 1, 5]> : tensor<3xi64>
    } : (tensor<5x1x5xi32>, tensor<i64>) -> tensor<1x5xi32>
    return %res : tensor<1x5xi32>
  }
  // CHECK 1x5xi32=[11 12 13 14 15]
}
