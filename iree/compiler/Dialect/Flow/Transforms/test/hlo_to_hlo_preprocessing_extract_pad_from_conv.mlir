// RUN: iree-opt -iree-flow-hlo-to-hlo-preprocessing -iree-flow-extract-pad-from-conv %s | IreeFileCheck %s

// CHECK-LABEL: @conv
//       CHECK: mhlo.pad
//  CHECK-SAME: edge_padding_high = dense<[0, 1, 1, 0]>
//  CHECK-SAME: edge_padding_low = dense<[0, 1, 0, 0]>
//       CHECK: mhlo.convolution
//   CHECK-NOT: padding
func @conv(%inputs: tensor<1x4x5x2xf32>, %weights: tensor<3x2x2x1xf32>) -> tensor<1x4x5x1xf32> {
  %0 = "mhlo.convolution"(%inputs, %weights) {
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
    output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
  feature_group_count = 1 : i64,
  padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>,
  rhs_dilation = dense<1> : tensor<2xi64>,
  window_strides = dense<1> : tensor<2xi64>} :
  (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x4x5x1xf32>
  return %0 : tensor<1x4x5x1xf32>
}
