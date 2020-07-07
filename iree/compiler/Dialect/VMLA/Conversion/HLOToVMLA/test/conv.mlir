// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @conv
func @conv(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32> attributes { sym_visibility = "private" } {
  // CHECK: vmla.conv
  // CHECK-SAME: {batch_group_count = 1 : i32,
  // CHECK-SAME: feature_group_count = 1 : i32,
  // CHECK-SAME: lhs_dilation = dense<1> : vector<2xi32>,
  // CHECK-SAME: padding = dense<[1, 2, 2, 2]> : vector<4xi32>,
  // CHECK-SAME: rhs_dilation = dense<1> : vector<2xi32>,
  // CHECK-SAME: window_strides = dense<1> : vector<2xi32>}
  %2 = "mhlo.convolution"(%arg0, %arg1) {
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
        rhs_dilation = dense<1> : tensor<2xi64>,
        lhs_dilation = dense<1> : tensor<2xi64>,
        padding = dense<[[1, 2],[2, 2]]> : tensor<2x2xi64>,
        window_strides = dense<1> : tensor<2xi64>} : (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
 return %2: tensor<1x2x3x1xf32>
}
