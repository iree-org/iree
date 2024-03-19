//===----------------------------------------------------------------------===//
// Pointwise convolution ops.
// Naming convention: '_'.join(
//   [conv,
//    {input-spatial-dims},
//    {output-spatial-dims},
//    {filter-dims},
//    {output-size x input-size}])
//===----------------------------------------------------------------------===//

func.func @conv_244_112_3x3_3x32() -> tensor<1x112x112x32xf32> {
    %input = util.unfoldable_constant dense<1.0> : tensor<1x224x224x3xf32>
    %filter = util.unfoldable_constant dense<1.0> : tensor<3x3x3x32xf32>
    %0 = "stablehlo.convolution"(%input, %filter) {
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
        padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 2, 2>
    } : (tensor<1x224x224x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32>
    return %0 : tensor<1x112x112x32xf32>
}

func.func @conv_112_112_1x1_32x64() -> tensor<1x112x112x64xf32> {
    %input = util.unfoldable_constant dense<1.0> : tensor<1x112x112x32xf32>
    %filter = util.unfoldable_constant dense<1.0> : tensor<1x1x32x64xf32>
    %0 = "stablehlo.convolution"(%input, %filter) {
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
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
    } : (tensor<1x112x112x32xf32>, tensor<1x1x32x64xf32>) -> tensor<1x112x112x64xf32>
    return %0 : tensor<1x112x112x64xf32>
}

func.func @conv_7_7_1x1_1024x1024() -> tensor<1x7x7x1024xf32> {
    %input = util.unfoldable_constant dense<1.0> : tensor<1x7x7x1024xf32>
    %filter = util.unfoldable_constant dense<1.0> : tensor<1x1x1024x1024xf32>
    %0 = "stablehlo.convolution"(%input, %filter) {
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
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
    } : (tensor<1x7x7x1024xf32>, tensor<1x1x1024x1024xf32>) -> tensor<1x7x7x1024xf32>
    return %0 : tensor<1x7x7x1024xf32>
}

//===----------------------------------------------------------------------===//
// Depthwise convolution ops.
// Naming convention: '_'.join(
//   [depthwise_conv,
//    {input-spatial-dim},
//    {output-spatial-dim},
//    {filter-size},
//    {output-size x input-size},
//    {feature-group}])
//===----------------------------------------------------------------------===//

func.func @depthwise_conv_15x1_1x1_15x1_1x1024_1024() -> tensor<1x1x1x1024xf32> {
  %input = util.unfoldable_constant dense<1.0> : tensor<1x15x1x1024xf32>
  %filter = util.unfoldable_constant dense<1.0> : tensor<15x1x1x1024xf32>
  %res = "stablehlo.convolution"(%input, %filter) {
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
    feature_group_count = 1024 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x15x1x1024xf32>, tensor<15x1x1x1024xf32>) -> tensor<1x1x1x1024xf32>
  return %res : tensor<1x1x1x1024xf32>
}

func.func @depthwise_conv_15x1_1x1_15x1_1x512_512() -> tensor<1x1x1x512xf32> {
  %input = util.unfoldable_constant dense<1.0> : tensor<1x15x1x512xf32>
  %filter = util.unfoldable_constant dense<1.0> : tensor<15x1x1x512xf32>
  %res = "stablehlo.convolution"(%input, %filter) {
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
    feature_group_count = 512 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x15x1x512xf32>, tensor<15x1x1x512xf32>) -> tensor<1x1x1x512xf32>
  return %res : tensor<1x1x1x512xf32>
}

func.func @depthwise_conv_16x1_2x1_16x1_1x512_512() -> tensor<1x2x1x512xf32> {
  %input = util.unfoldable_constant dense<1.0> : tensor<1x16x1x512xf32>
  %filter = util.unfoldable_constant dense<1.0> : tensor<15x1x1x512xf32>
  %res = "stablehlo.convolution"(%input, %filter) {
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
    feature_group_count = 512 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x16x1x512xf32>, tensor<15x1x1x512xf32>) -> tensor<1x2x1x512xf32>
  return %res : tensor<1x2x1x512xf32>
}
