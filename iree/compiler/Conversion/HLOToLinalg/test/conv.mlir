// RUN: iree-opt -iree-codegen-hlo-to-linalg-on-tensors -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: func @linalg.conv_1d_input_nwc_filter_wcf
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = constant 2 : index
// CHECK-DAG:     %[[ZERO:.+]] = constant 0.000000e+00 : f32
// CHECK:         %[[DIM0:.+]] = dim %[[ARG0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[DIM1:.+]] = dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[DIM2:.+]] = dim %[[ARG1]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK:         %[[FILL:.+]] = linalg.fill(%[[INIT]], %[[ZERO]])
// CHECK:         linalg.conv_1d_input_nwc_filter_wcf
// CHECK-SAME:      {dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:       strides = dense<1> : tensor<1xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
func @linalg.conv_1d_input_nwc_filter_wcf(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>)
  -> tensor<?x?x?xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = {
      input_batch_dimension = 0 : i64,
      input_feature_dimension = 2 : i64,
      input_spatial_dimensions = dense<[1]> : tensor<1xi64>,
      kernel_input_feature_dimension = 1 : i64,
      kernel_output_feature_dimension = 2 : i64,
      kernel_spatial_dimensions = dense<[0]> : tensor<1xi64>,
      output_batch_dimension = 0 : i64,
      output_feature_dimension = 2 : i64,
      output_spatial_dimensions = dense<[1]> : tensor<1xi64>
    },
    feature_group_count = 1 : i64,
    padding = dense<[[0], [0]]> : tensor<2x1xi64>,
    rhs_dilation = dense<1> : tensor<1xi64>,
    window_strides = dense<1> : tensor<1xi64>
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func @conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = constant 2 : index
// CHECK-DAG:     %[[C3:.+]] = constant 3 : index
// CHECK-DAG:     %[[ZERO:.+]] = constant 0.000000e+00 : f32
// CHECK:         %[[DIM0:.+]] = dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[DIM1:.+]] = dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[DIM2:.+]] = dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[DIM3:.+]] = dim %[[ARG1]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]]]
// CHECK:         %[[FILL:.+]] = linalg.fill(%[[INIT]], %[[ZERO]])
// CHECK:         linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:      {dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:       strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
func @conv_2d_input_nhwc_filter_hwcf(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>)
  -> tensor<?x?x?x?xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
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
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func @conv_3d_input_ndhwc_filter_dhwcf
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = constant 2 : index
// CHECK-DAG:     %[[C3:.+]] = constant 3 : index
// CHECK-DAG:     %[[C4:.+]] = constant 4 : index
// CHECK-DAG:     %[[ZERO:.+]] = constant 0.000000e+00 : f32
// CHECK:         %[[DIM0:.+]] = dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[DIM1:.+]] = dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[DIM2:.+]] = dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[DIM3:.+]] = dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[DIM4:.+]] = dim %[[ARG1]], %[[C4]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]], %[[DIM4]]]
// CHECK:         %[[FILL:.+]] = linalg.fill(%[[INIT]], %[[ZERO]])
// CHECK:         linalg.conv_3d_input_ndhwc_filter_dhwcf
// CHECK-SAME:      {dilations = dense<1> : tensor<3xi64>
// CHECK-SAME:       strides = dense<1> : tensor<3xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
func @conv_3d_input_ndhwc_filter_dhwcf(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>)
  -> tensor<?x?x?x?x?xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = {
      input_batch_dimension = 0 : i64,
      input_feature_dimension = 4 : i64,
      input_spatial_dimensions = dense<[1, 2, 3]> : tensor<3xi64>,
      kernel_input_feature_dimension = 3 : i64,
      kernel_output_feature_dimension = 4 : i64,
      kernel_spatial_dimensions = dense<[0, 1, 2]> : tensor<3xi64>,
      output_batch_dimension = 0 : i64,
      output_feature_dimension = 4 : i64,
      output_spatial_dimensions = dense<[1, 2, 3]> : tensor<3xi64>
    },
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0, 0], [0, 0, 0]]> : tensor<2x3xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>,
    window_strides = dense<1> : tensor<3xi64>
  } : (tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// CHECK-LABEL: func @conv2d_1452x2223_dilated_valid
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[ZERO:.+]] = constant 0.000000e+00 : f32
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 2, 4, 3] : tensor<1x2x4x3xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill(%[[INIT]], %[[ZERO]]) : tensor<1x2x4x3xf32>, f32 -> tensor<1x2x4x3xf32>
// CHECK:         linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:      {dilations = dense<[2, 1]> : tensor<2xi64>
// CHECK-SAME:       strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<1x4x5x2xf32>, tensor<2x2x2x3xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
func @conv2d_1452x2223_dilated_valid(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<2x2x2x3xf32>)
  -> tensor<1x2x4x3xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
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
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<[2, 1]> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x4x5x2xf32>, tensor<2x2x2x3xf32>) -> tensor<1x2x4x3xf32>
  return %0 : tensor<1x2x4x3xf32>
}
