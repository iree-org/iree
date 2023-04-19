// RUN: iree-opt %s --iree-stablehlo-to-linalg --split-input-file \
// RUN:   --canonicalize | FileCheck %s

// CHECK-LABEL: @linalg.conv_0d_nc
func.func @linalg.conv_0d_nc(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<3x3xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, f]x[i, o]->[b, f],
         window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []}
         {
           batch_group_count = 1 : i64, feature_group_count = 1 : i64,
           precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
         } : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00
// CHECK-DAG: %[[INIT:.+]] = tensor.empty()
// CHECK-DAG: %[[FILL:.+]] = linalg.fill ins(%cst{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%[[FILL]] : tensor<3x3xf32>)

// -----

// CHECK-LABEL: func @linalg.conv_1d_nwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
func.func @linalg.conv_1d_nwc(%arg0: tensor<?x8x?xf32>, %arg1: tensor<2x?x?xf32>)
  -> tensor<?x7x?xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 2,
      input_spatial_dimensions = [1],
      kernel_input_feature_dimension = 1,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0],
      output_batch_dimension = 0,
      output_feature_dimension = 2,
      output_spatial_dimensions = [1]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0]]> : tensor<1x2xi64>,
    rhs_dilation = dense<1> : tensor<1xi64>,
    window_strides = dense<1> : tensor<1xi64>,
    someattr
  } : (tensor<?x8x?xf32>, tensor<2x?x?xf32>) -> tensor<?x7x?xf32>
  func.return %0 : tensor<?x7x?xf32>
}
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x8x?xf32>
// CHECK:         %[[DIM2:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<2x?x?xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[DIM0]], %[[DIM2]])
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK:         linalg.conv_1d_nwc_wcf
// CHECK-SAME:      {dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:       someattr
// CHECK-SAME:       strides = dense<1> : tensor<1xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x8x?xf32>, tensor<2x?x?xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<?x7x?xf32>) -> tensor<?x7x?xf32>

// -----

// CHECK-LABEL: func @conv_2d_nhwc_hwcf
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @conv_2d_nhwc_hwcf(%arg0: tensor<?x4x5x?xf32>, %arg1: tensor<3x2x?x?xf32>)
  -> tensor<?x2x4x?xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
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
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<?x4x5x?xf32>, tensor<3x2x?x?xf32>) -> tensor<?x2x4x?xf32>
  func.return %0 : tensor<?x2x4x?xf32>
}
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x4x5x?xf32>
// CHECK:         %[[DIM3:.+]] = tensor.dim %[[ARG1]], %[[C3]] : tensor<3x2x?x?xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[DIM0]], %[[DIM3]])
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK:         linalg.conv_2d_nhwc
// CHECK-SAME:      {dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:       strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x4x5x?xf32>, tensor<3x2x?x?xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<?x2x4x?xf32>) -> tensor<?x2x4x?xf32>

// -----

// CHECK-LABEL: func @conv_transpose_2d
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @conv_transpose_2d(%arg0: tensor<2x9x10x3xf32>,
                             %arg1: tensor<4x4x3x3xf32>)
  -> tensor<2x15x25x3xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[6, 6], [6, 6]],
              lhs_dilate = [1, 2], rhs_dilate = [2, 2]}
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    } : (tensor<2x9x10x3xf32>, tensor<4x4x3x3xf32>) -> tensor<2x15x25x3xf32>
  return %0 : tensor<2x15x25x3xf32>
}
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[INIT:.+]] = tensor.empty()
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK:         %[[LHS_INIT:.+]] = tensor.empty()
// CHECK:         %[[LHS_FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[LHS_INIT]]
// CHECK:         %[[LHS_PAD:.+]] = tensor.insert_slice %[[ARG0]] into %[[LHS_FILL]][0, 6, 6, 0] [2, 9, 10, 3] [1, 1, 2, 1] : tensor<2x9x10x3xf32> into tensor<2x21x31x3xf32>
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      {dilations = dense<2> : tensor<2xi64>
// CHECK-SAME:       strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[LHS_PAD]], %[[ARG1]] : tensor<2x21x31x3xf32>, tensor<4x4x3x3xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<2x15x25x3xf32>) -> tensor<2x15x25x3xf32>

// -----

// Just check that this lowers successfully.
// CHECK-LABEL: func @conv_different_batch_dim_in_out
func.func @conv_different_batch_dim_in_out(%arg0: tensor<1x1x1xf64>,
                                           %arg1: tensor<1x1x1xf64>)
  -> tensor<1x1x1xf64> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [f, 0, b]x[i, o, 0]->[f, b, 0],
    window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1],
             rhs_dilate = [1]}
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]
    } : (tensor<1x1x1xf64>, tensor<1x1x1xf64>) -> tensor<1x1x1xf64>
  return %0 : tensor<1x1x1xf64>
}

// -----

// Just check that this lowers successfully.
// CHECK-LABEL: func @conv_different_batch_dim_in_out_with_feature_group_count
func.func @conv_different_batch_dim_in_out_with_feature_group_count(
    %arg0: tensor<4x6x7x1xf64>, %arg1: tensor<2x6x3x2xf64>)
  -> tensor<1x2x1x2xf64> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, -1]],
              lhs_dilate = [1, 1], rhs_dilate = [1, 2],
              reverse = [0, 0]}
    {
      batch_group_count = 1 : i64,
      feature_group_count = 2 : i64,
      precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]
    } : (tensor<4x6x7x1xf64>, tensor<2x6x3x2xf64>) -> tensor<1x2x1x2xf64>
  return %0 : tensor<1x2x1x2xf64>
}

// -----

// CHECK-LABEL: func @conv_3d_ndhwc_dhwcf
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @conv_3d_ndhwc_dhwcf(%arg0: tensor<?x8x8x8x?xf32>, %arg1: tensor<2x2x2x?x?xf32>)
  -> tensor<?x7x7x7x?xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 4,
      input_spatial_dimensions = [1, 2, 3],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 4,
      kernel_spatial_dimensions = [0, 1, 2],
      output_batch_dimension = 0,
      output_feature_dimension = 4,
      output_spatial_dimensions = [1, 2, 3]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>,
    window_strides = dense<1> : tensor<3xi64>
  } : (tensor<?x8x8x8x?xf32>, tensor<2x2x2x?x?xf32>) -> tensor<?x7x7x7x?xf32>
  func.return %0 : tensor<?x7x7x7x?xf32>
}
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x8x8x8x?xf32>
// CHECK:         %[[DIM4:.+]] = tensor.dim %[[ARG1]], %[[C4]] : tensor<2x2x2x?x?xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[DIM0]], %[[DIM4]])
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK:         linalg.conv_3d_ndhwc_dhwcf
// CHECK-SAME:      {dilations = dense<1> : tensor<3xi64>
// CHECK-SAME:       strides = dense<1> : tensor<3xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x8x8x8x?xf32>, tensor<2x2x2x?x?xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<?x7x7x7x?xf32>) -> tensor<?x7x7x7x?xf32>

// -----

// CHECK-LABEL: func @conv2d_1452x2223_dilated_valid
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @conv2d_1452x2223_dilated_valid(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<2x2x2x3xf32>)
  -> tensor<1x2x4x3xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
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
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<[2, 1]> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x4x5x2xf32>, tensor<2x2x2x3xf32>) -> tensor<1x2x4x3xf32>
  func.return %0 : tensor<1x2x4x3xf32>
}
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x2x4x3xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[INIT]] : tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      {dilations = dense<[2, 1]> : tensor<2xi64>
// CHECK-SAME:       strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<1x4x5x2xf32>, tensor<2x2x2x3xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>

// -----

// CHECK-LABEL: func @linalg.conv_2D_padding_test1
// CHECK-SAME: (%[[FILTER:.*]]: tensor<1x33x1x1xf16>, %[[INPUT:.*]]: tensor<400x1024x1024x1xf16>)
func.func @linalg.conv_2D_padding_test1(%arg0: tensor<1x33x1x1xf16>, %arg1: tensor<400x1024x1024x1xf16>)
  -> tensor<400x1024x1024x1xf16> {
  %0 = stablehlo.convolution(%arg1, %arg0)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = { stride = [1, 1], pad = [[0, 0], [16, 16]], rhs_dilate = [1, 1] }
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64
         } : (tensor<400x1024x1024x1xf16>, tensor<1x33x1x1xf16>) -> (tensor<400x1024x1024x1xf16>)
  func.return %0 : tensor<400x1024x1024x1xf16>
}
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[INIT:.*]] = tensor.empty() : tensor<400x1024x1024x1xf16>
// CHECK-NEXT: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[INIT]] : tensor<400x1024x1024x1xf16>) -> tensor<400x1024x1024x1xf16>
// CHECK-NEXT: %[[PAD:.*]] = tensor.pad %[[INPUT]] low[0, 0, 16, 0] high[0, 0, 16, 0]  {
// CHECK-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK-NEXT:   tensor.yield %[[ZERO]] : f16
// CHECK-NEXT: } : tensor<400x1024x1024x1xf16> to tensor<400x1024x1056x1xf16>
// CHECK-NEXT: %[[RESULT:.*]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%[[PAD]], %[[FILTER]] : tensor<400x1024x1056x1xf16>, tensor<1x33x1x1xf16>) outs(%[[FILL]] : tensor<400x1024x1024x1xf16>) -> tensor<400x1024x1024x1xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<400x1024x1024x1xf16>

// -----

// CHECK-LABEL: func @linalg.conv_2D_padding_test2
// CHECK-SAME: (%[[FILTER:.*]]: tensor<1x33x1x1xf16>, %[[INPUT:.*]]: tensor<400x1024x1024x1xf16>)
func.func @linalg.conv_2D_padding_test2(%arg0: tensor<1x33x1x1xf16>, %arg1: tensor<400x1024x1024x1xf16>)
  -> tensor<400x1040x1024x1xf16> {
  %0 = stablehlo.convolution(%arg1, %arg0)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[8, 8], [16, 16]], rhs_dilate = [1, 1]}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64
         } : (tensor<400x1024x1024x1xf16>, tensor<1x33x1x1xf16>) -> (tensor<400x1040x1024x1xf16>)
  return %0 : tensor<400x1040x1024x1xf16>
}
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<400x1040x1024x1xf16>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[INIT]] : tensor<400x1040x1024x1xf16>) -> tensor<400x1040x1024x1xf16>
// CHECK-NEXT: %[[PAD:.*]] = tensor.pad %[[INPUT]] low[0, 8, 16, 0] high[0, 8, 16, 0]  {
// CHECK-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK-NEXT:   tensor.yield %[[ZERO]] : f16
// CHECK-NEXT: } : tensor<400x1024x1024x1xf16> to tensor<400x1040x1056x1xf16>
// CHECK-NEXT: %[[RESULT:.*]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%[[PAD]], %arg0 : tensor<400x1040x1056x1xf16>, tensor<1x33x1x1xf16>) outs(%[[FILL]] : tensor<400x1040x1024x1xf16>) -> tensor<400x1040x1024x1xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<400x1040x1024x1xf16>

// -----

// CHECK-LABEL:  func @depthwise_conv
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]
func.func @depthwise_conv(%arg0: tensor<2x4x5x2xf32>,
                     %arg1: tensor<2x2x1x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
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
    feature_group_count = 2 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>,
    someattr} : (tensor<2x4x5x2xf32>, tensor<2x2x1x6xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>
}
// CHECK-DAG:       %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[COLLAPSE:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2, 3]] : tensor<2x2x1x6xf32> into tensor<24xf32>
// CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]] {{\[}}[0, 1, 2, 3]] : tensor<24xf32> into tensor<2x2x2x3xf32>
// CHECK:       %[[INIT:.+]] = tensor.empty() : tensor<2x3x4x2x3xf32>
// CHECK:       %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
// CHECK:       %[[OUT:.+]] = linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-SAME:     {dilations = dense<1> : tensor<2xi64>, someattr, strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[IN]], %[[EXPAND]] : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
// CHECK:       %{{.+}} = tensor.collapse_shape %[[OUT]]
// CHECK-SAME:     [0], [1], [2], [3, 4]
// CHECK-SAME:     : tensor<2x3x4x2x3xf32> into tensor<2x3x4x6xf32>

// -----

// CHECK-LABEL:  func @depthwise_conv_with_padding
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]
func.func @depthwise_conv_with_padding(
    %arg0: tensor<2x4x5x2xf32>,
    %arg1: tensor<2x2x1x4xf32>) -> tensor<2x3x6x4xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
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
    feature_group_count = 2 : i64,
    padding = dense<[[0, 0], [1, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>,
    someattr} : (tensor<2x4x5x2xf32>, tensor<2x2x1x4xf32>) -> tensor<2x3x6x4xf32>
  func.return %0 : tensor<2x3x6x4xf32>
}
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[PAD:.*]] = tensor.pad %[[IN]] low[0, 0, 1, 0] high[0, 0, 1, 0] {
// CHECK:        ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK:          tensor.yield %[[ZERO]] : f32
// CHECK         } : tensor<2x4x5x2xf32> to tensor<2x4x7x2xf32>
// CHECK:        %[[COLLAPSE:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:    [0, 1, 2, 3]
// CHECK-SAME:    : tensor<2x2x1x4xf32> into tensor<16xf32>
// CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]]
// CHECK-SAME:   [0, 1, 2, 3]
// CHECK-SAME:   tensor<16xf32> into tensor<2x2x2x2xf32>
// CHECK:        %[[INIT:.+]] = tensor.empty() : tensor<2x3x6x2x2xf32>
// CHECK:        %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[INIT]] : tensor<2x3x6x2x2xf32>) -> tensor<2x3x6x2x2xf32>
// CHECK:        %[[OUT:.+]] = linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-SAME:     {dilations = dense<1> : tensor<2xi64>, someattr, strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[PAD]], %[[EXPAND]] : tensor<2x4x7x2xf32>, tensor<2x2x2x2xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<2x3x6x2x2xf32>) -> tensor<2x3x6x2x2xf32>
// CHECK:        %{{.+}} = tensor.collapse_shape %[[OUT]]
// CHECK-SAME:     [0], [1], [2], [3, 4]
// CHECK-SAME:     : tensor<2x3x6x2x2xf32> into tensor<2x3x6x4xf32>

// -----

// CHECK-LABEL:   func @depthwise_conv_multiplier_1
// CHECK-SAME:    %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9_]*]]
func.func @depthwise_conv_multiplier_1(%arg0: tensor<1x113x113x96xf32>,
                                  %arg1: tensor<3x3x1x96xf32>) -> tensor<1x56x56x96xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
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
    feature_group_count = 96 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<2> : tensor<2xi64>} : (tensor<1x113x113x96xf32>, tensor<3x3x1x96xf32>) -> tensor<1x56x56x96xf32>
  func.return %0 : tensor<1x56x56x96xf32>
}
// CHECK-DAG:     %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x56x56x96xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
// CHECK:         %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:     [0], [1], [2, 3]
// CHECK-SAME:     : tensor<3x3x1x96xf32> into tensor<3x3x96xf32>
// CHECK:         %{{.+}} = linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
// CHECK-SAME:       ins(%[[IN]], %[[RESHAPED_FILTER]] : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>

// -----

// CHECK-LABEL:   func @depthwise_conv_multiplier_1_with_padding
// CHECK-SAME:    %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9_]*]]
func.func @depthwise_conv_multiplier_1_with_padding(
    %arg0: tensor<1x113x113x96xf32>,
    %arg1: tensor<3x3x1x96xf32>) -> tensor<1x57x58x96xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
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
    feature_group_count = 96 : i64,
    padding = dense<[[1, 1], [2, 2]]> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<2> : tensor<2xi64>} : (tensor<1x113x113x96xf32>, tensor<3x3x1x96xf32>) -> tensor<1x57x58x96xf32>
  func.return %0 : tensor<1x57x58x96xf32>
}
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[PAD:.*]] = tensor.pad %[[IN]] low[0, 1, 2, 0] high[0, 1, 2, 0]  {
// CHECK:         ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK          } : tensor<1x113x113x96xf32> to tensor<1x115x117x96xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x57x58x96xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[INIT]] : tensor<1x57x58x96xf32>) -> tensor<1x57x58x96xf32>
// CHECK:         %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:     [0], [1], [2, 3]
// CHECK-SAME:     : tensor<3x3x1x96xf32> into tensor<3x3x96xf32>
// CHECK:         %{{.+}} = linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
// CHECK-SAME:       ins(%[[PAD]], %[[RESHAPED_FILTER]] : tensor<1x115x117x96xf32>, tensor<3x3x96xf32>)
// CHECK-SAME:       outs(%[[FILL]] : tensor<1x57x58x96xf32>) -> tensor<1x57x58x96xf32>

// -----

// CHECK-LABEL:  func @depthwise_conv1d
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]
func.func @depthwise_conv1d(%arg0: tensor<1x10x8xf32>,
                            %arg1: tensor<3x1x16xf32>) -> tensor<1x10x16xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {
      stride = [1],
      pad = [[1, 1]],
      lhs_dilate = [1],
      rhs_dilate = [1],
      reverse = [0]} {
    batch_group_count = 1 : i64,
    feature_group_count = 8 : i64,
    someattr} : (tensor<1x10x8xf32>, tensor<3x1x16xf32>) -> tensor<1x10x16xf32>
  func.return %0 : tensor<1x10x16xf32>
}
// CHECK:       %[[CONV:.+]] = linalg.depthwise_conv_1d_nwc_wcm
// CHECK:       %[[OUT:.+]] = tensor.collapse_shape %[[CONV]]
// CHECK:       return %[[OUT]]

// -----

// CHECK-LABEL:  func @depthwise_conv1d
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]
func.func @depthwise_conv1d_m1(%arg0: tensor<1x10x8xf32>,
                               %arg1: tensor<3x1x8xf32>) -> tensor<1x10x8xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {
      stride = [1],
      pad = [[1, 1]],
      lhs_dilate = [1],
      rhs_dilate = [1],
      reverse = [0]} {
    batch_group_count = 1 : i64,
    feature_group_count = 8 : i64,
    someattr} : (tensor<1x10x8xf32>, tensor<3x1x8xf32>) -> tensor<1x10x8xf32>
  func.return %0 : tensor<1x10x8xf32>
}
// CHECK:       %[[CONV:.+]] = linalg.depthwise_conv_1d_nwc_wc
// CHECK:       return %[[CONV]]

// -----

// CHECK-LABEL:  func @depthwise_conv3d
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]
func.func @depthwise_conv3d(%arg0: tensor<2x3x5x4x6xf32>,
                            %arg1: tensor<2x1x3x1x36xf32>)
                            -> tensor<2x3x13x4x36xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [2, 1, 3],
      pad = [[1, 2], [5, 3], [3, 5]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1],
      reverse = [0, 0, 0]} {
    batch_group_count = 1 : i64,
    feature_group_count = 6 : i64,
    someattr} : (tensor<2x3x5x4x6xf32>, tensor<2x1x3x1x36xf32>)
              -> tensor<2x3x13x4x36xf32>
  func.return %0 : tensor<2x3x13x4x36xf32>
}
// CHECK:       %[[CONV:.+]] = linalg.depthwise_conv_3d_ndhwc_dhwcm
// CHECK:       %[[OUT:.+]] = tensor.collapse_shape %[[CONV]]
// CHECK:       return %[[OUT]]

// -----

// CHECK-LABEL:  func @depthwise_conv3d
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]
func.func @depthwise_conv3d_m1(%arg0: tensor<2x3x5x4x6xf32>,
                               %arg1: tensor<2x1x3x1x6xf32>)
                               -> tensor<2x3x13x4x6xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [2, 1, 3],
      pad = [[1, 2], [5, 3], [3, 5]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1],
      reverse = [0, 0, 0]} {
    batch_group_count = 1 : i64,
    feature_group_count = 6 : i64,
    someattr} : (tensor<2x3x5x4x6xf32>, tensor<2x1x3x1x6xf32>)
              -> tensor<2x3x13x4x6xf32>
  func.return %0 : tensor<2x3x13x4x6xf32>
}
// CHECK:       %[[CONV:.+]] = linalg.depthwise_conv_3d_ndhwc_dhwc
// CHECK:       return %[[CONV]]
