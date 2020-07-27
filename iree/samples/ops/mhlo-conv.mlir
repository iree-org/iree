func @conv(%lhs: tensor<1x9x9x512xf32>, %rhs: tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32> attributes {iree.module.export} {
  %0 = "mhlo.convolution"(%lhs, %rhs) {
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
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
  return %0 : tensor<1x7x7x512xf32>
}
