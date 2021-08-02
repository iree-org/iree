// RUN: iree-opt -split-input-file -iree-flow-change-conv2d-layout %s | IreeFileCheck %s

//  CHECK-LABEL: func @conv_filter_fhwc_to_hwcf
//   CHECK-SAME: (%[[INPUT:.+]]: tensor<1x2x5x4xf32>, %[[INIT:.+]]: tensor<1x2x2x2xf32>)
func @conv_filter_fhwc_to_hwcf(%input: tensor<1x2x5x4xf32>, %init: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
  //      CHECK: %[[FILTER:.+]] = constant
  // CHECK-SAME{LITERAL}: dense<[[
  // CHECK-SAME{LITERAL}:   [[1.000000e+00, 1.300000e+01], [2.000000e+00, 1.400000e+01], [3.000000e+00, 1.500000e+01], [4.000000e+00, 1.600000e+01]],
  // CHECK-SAME{LITERAL}:   [[5.000000e+00, 1.700000e+01], [6.000000e+00, 1.800000e+01], [7.000000e+00, 1.900000e+01], [8.000000e+00, 2.000000e+01]],
  // CHECK-SAME{LITERAL}:   [[9.000000e+00, 2.100000e+01], [1.000000e+01, 2.200000e+01], [1.100000e+01, 2.300000e+01], [1.200000e+01, 2.400000e+01]]
  // CHECK-SAME{LITERAL}: ]]> : tensor<1x3x4x2xf32>
  %filter = constant dense<[
    [[[ 1.0,  2.0,  3.0,  4.0], [ 5.0,  6.0,  7.0,  8.0], [ 9.0, 10.0, 11.0, 12.0]]],
    [[[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]
  ]> : tensor<2x1x3x4xf32>
  //      CHECK: %[[CONV:.+]] = linalg.conv_2d_input_nhwc_filter_hwcf
  // CHECK-SAME:   {dilations = dense<1> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>}
  // CHECK-SAME: ins(%[[INPUT]], %[[FILTER]] : tensor<1x2x5x4xf32>, tensor<1x3x4x2xf32>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<1x2x2x2xf32>)
  %0 = linalg.conv_2d_input_nhwc_filter_ohwi_poly
         {dilations = dense<1> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>}
       ins(%input, %filter : tensor<1x2x5x4xf32>, tensor<2x1x3x4xf32>)
       outs(%init : tensor<1x2x2x2xf32>)
       -> tensor<1x2x2x2xf32>
  //      CHECK: return %[[CONV]]
  return %0: tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func @dont_change_conv_without_const_filter
func @dont_change_conv_without_const_filter(%input: tensor<1x2x5x4xf32>, %filter: tensor<2x1x3x4xf32>, %init: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
  // CHECK: linalg.conv_2d_input_nhwc_filter_ohwi_poly
  %0 = linalg.conv_2d_input_nhwc_filter_ohwi_poly
         {dilations = dense<1> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>}
       ins(%input, %filter : tensor<1x2x5x4xf32>, tensor<2x1x3x4xf32>)
       outs(%init : tensor<1x2x2x2xf32>)
       -> tensor<1x2x2x2xf32>
  return %0: tensor<1x2x2x2xf32>
}
