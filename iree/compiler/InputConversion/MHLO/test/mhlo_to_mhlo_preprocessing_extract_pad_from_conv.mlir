// RUN: iree-opt -iree-mhlo-to-mhlo-preprocessing %s | IreeFileCheck %s

// CHECK-LABEL: @conv
//       CHECK: mhlo.pad
//  CHECK-SAME: edge_padding_high = dense<[0, 1, 1, 0]>
//  CHECK-SAME: edge_padding_low = dense<[0, 1, 0, 0]>
//       CHECK: mhlo.convolution
//   CHECK-NOT: padding
func @conv(%inputs: tensor<1x4x5x2xf32>, %weights: tensor<3x2x2x1xf32>) -> tensor<1x4x5x1xf32> {
  %0 = "mhlo.convolution"(%inputs, %weights) {
  batch_group_count = 1 : i64,
  dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
  feature_group_count = 1 : i64,
  padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>,
  rhs_dilation = dense<1> : tensor<2xi64>,
  window_strides = dense<1> : tensor<2xi64>} :
  (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x4x5x1xf32>
  return %0 : tensor<1x4x5x1xf32>
}
