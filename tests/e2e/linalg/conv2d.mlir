func.func @conv2d_nopadding() {
  %inputs = util.unfoldable_constant dense<[[[
         [1.0, 3.0, 5.0, 7.0],
         [11.0, 13.0, 15.0, 17.0],
         [21.0, 23.0, 25.0, 27.0],
         [31.0, 33.0, 35.0, 37.0]],
        [[2.0, 4.0, 6.0, 8.0],
         [12.0, 14.0, 16.0, 18.0],
         [22.0, 24.0, 26.0, 28.0],
         [32.0, 34.0, 36.0, 38.0]]]]> : tensor<1x2x4x4xf32>
  %weights = util.unfoldable_constant dense<[[
        [[1.0, 3.0],
         [5.0, 7.0],
         [9.0, 11.0]],
        [[2.0, 4.0],
         [6.0, 8.0],
         [10.0, 12.0]]]]> : tensor<1x2x3x2xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = tensor.empty() : tensor<1x1x2x3xf32>
  %out = linalg.fill ins(%cst : f32) outs(%fill : tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32>
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%inputs, %weights : tensor<1x2x4x4xf32>, tensor<1x2x3x2xf32>) outs(%out : tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32>
  check.expect_almost_eq_const(%result, dense<[[
        [[1310.0, 1466.0, 1622.0],
         [2090.0, 2246.0, 2402.0]]
  ]]> : tensor<1x1x2x3xf32>) : tensor<1x1x2x3xf32>
  return
}
