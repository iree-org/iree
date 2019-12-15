// RUN: iree-run-mlir --target_backends=interpreter-bytecode --input_values="1x2xf32= 1 2\n2x1x4xf32= 5 6 7 8 9 10 11 12\ni64= 0" --output_types=f %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --target_backends=vulkan-spirv --input_values="1x2xf32= 1 2\n2x1x4xf32= 5 6 7 8 9 10 11 12\ni64= 0" --output_types=f %s | IreeFileCheck %s)

module {
  func @gather_concat(%arg0: tensor<1x2xf32>, %arg1: tensor<2x1x4xf32>, %arg2: tensor<i64>) -> tensor<1x6xf32> {
    %0 = "xla_hlo.gather"(%arg1, %arg2) {dimension_numbers = {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, start_index_map = dense<0> : tensor<1xi64>}, slice_sizes = dense<[1, 1, 4]> : tensor<3xi64>} : (tensor<2x1x4xf32>, tensor<i64>) -> tensor<1x4xf32>
    %1 = "xla_hlo.concatenate"(%0, %arg0) {dimension = 1 : i64} : (tensor<1x4xf32>, tensor<1x2xf32>) -> tensor<1x6xf32>
    return %1 : tensor<1x6xf32>
  }
  // CHECK: 1x6xf32=[5 6 7 8 1 2]
}
