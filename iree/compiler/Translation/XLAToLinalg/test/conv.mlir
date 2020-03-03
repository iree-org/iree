// RUN: iree-opt -iree-hlo-to-named-linalg %s | IreeFileCheck %s

// CHECK: func @conv
func @conv(%arg0: memref<3x5x5x3xf32>, %arg1: memref<2x2x3x4xf32>, %arg2: memref<3x5x5x4xf32>) {
  %0 = iree.load_input(%arg0 : memref<3x5x5x3xf32>) : tensor<3x5x5x3xf32>
  %1 = iree.load_input(%arg1 : memref<2x2x3x4xf32>) : tensor<2x2x3x4xf32>
  // CHECK: linalg.conv(%arg0, %arg1, %arg2) {dilations = [1, 2], strides = [2, 1]}
  %2 = "xla_hlo.conv"(%1, %0) {
    batch_group_count = 1 : i64,
    dimension_numbers = {
      input_batch_dimension = 0 : i64,
      input_feature_dimension = 3 : i64,
      input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
      kernel_input_feature_dimension = 2 : i64,
      kernel_output_feature_dimension = 3 : i64,
      kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
      output_batch_dimension = 0 : i64,
      output_feature_dimension = 3 : i64,
      output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
    },
    feature_group_count = 1 : i64,
    padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<[1, 2]> : tensor<2xi64>,
    window_strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
  iree.store_output(%2 : tensor<3x5x5x4xf32>, %arg2 : memref<3x5x5x4xf32>)
  return
}
