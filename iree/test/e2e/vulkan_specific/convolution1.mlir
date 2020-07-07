func @conv2d_nopadding() attributes { iree.module.export } {
  %inputs = iree.unfoldable_constant dense<[[
      [[ 1.0,  2.0], [ 3.0,  4.0], [ 5.0,  6.0], [ 7.0,  8.0], [ 9.0, 10.0]],
      [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
      [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0], [29.0, 30.0]],
      [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [39.0, 40.0]]]]> : tensor<1x4x5x2xf32>
  %weights = iree.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0]], [[ 3.0], [ 4.0]]],
      [[[ 5.0], [ 6.0]], [[ 7.0], [ 8.0]]],
      [[[ 9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %res = "mhlo.convolution"(%inputs, %weights) {
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
        window_strides = dense<1> : tensor<2xi64>} : (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
  check.expect_almost_eq_const(%res, dense<[[
      [[1310.0],[1466.0],[1622.0]],
      [[2090.0],[2246.0],[2402.0]]
  ]]> : tensor<1x2x3x1xf32>) : tensor<1x2x3x1xf32>
  return
}

func @conv2d_1452x3221_same() attributes { iree.module.export } {
  %inputs = iree.unfoldable_constant dense<[[
      [[ 1.0,  2.0], [ 3.0,  4.0], [ 5.0,  6.0], [ 7.0,  8.0], [ 9.0, 10.0]],
      [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
      [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0], [29.0, 30.0]],
      [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [39.0, 40.0]]]]> : tensor<1x4x5x2xf32>
  %weights = iree.unfoldable_constant dense<[
      [[[ 1.0], [ 2.0]], [[ 3.0], [ 4.0]]],
      [[[ 5.0], [ 6.0]], [[ 7.0], [ 8.0]]],
      [[[ 9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %res = "mhlo.convolution"(%inputs, %weights) {
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
  check.expect_almost_eq_const(%res,  dense<[[
    [[ 600.0], [ 736.0], [ 872.0], [1008.0], [ 476.0]],
    [[1310.0], [1466.0], [1622.0], [1778.0], [ 805.0]],
    [[2090.0], [2246.0], [2402.0], [2558.0], [1135.0]],
    [[1080.0], [1152.0], [1224.0], [1296.0], [ 524.0]]]]> : tensor<1x4x5x1xf32>) : tensor<1x4x5x1xf32>
  return
}
