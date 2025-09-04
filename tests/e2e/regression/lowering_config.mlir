#compilation0 = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32], [1, 8, 0], [0, 0, 8], [0, 0, 0]]>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>>
#compilation1 = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [1, 4, 0], [0, 0, 4], [0, 0, 0]]>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>>
#compilation2 = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [{sizes=[32, 64], interchange=[1, 0]}, [1, 1, 0], [0, 0, 8], [0, 0, 0]]>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>>

func.func @lowering_config_test() {
  %a = util.unfoldable_constant dense<1.0> : tensor<128x256xf32>
  %b = util.unfoldable_constant dense<2.0> : tensor<256x512xf32>
  %c = util.unfoldable_constant dense<2.0> : tensor<256x1024xf32>
  %0 = "stablehlo.dot"(%a, %b) {compilation_info = #compilation0} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  %1 = "stablehlo.dot"(%a, %c) {compilation_info = #compilation1} : (tensor<128x256xf32>, tensor<256x1024xf32>) -> tensor<128x1024xf32>
  %2 = "stablehlo.dot"(%a, %c) {compilation_info = #compilation2} : (tensor<128x256xf32>, tensor<256x1024xf32>) -> tensor<128x1024xf32>
  check.expect_almost_eq_const(%0, dense<512.0> : tensor<128x512xf32>) : tensor<128x512xf32>
  check.expect_almost_eq_const(%1, dense<512.0> : tensor<128x1024xf32>) : tensor<128x1024xf32>
  check.expect_almost_eq_const(%2, dense<512.0> : tensor<128x1024xf32>) : tensor<128x1024xf32>
  return
}

// Conv dims: N, OH, OW, OC, KH, KW, (IC)
// Remove H
#conv_compilation0 = #iree_codegen.compilation_info<
    lowering_config = #iree_cpu.lowering_config<distribution = [0, 7, 7, 64, 0, 0, 0], vector_common_parallel = [1, 1, 7, 4, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3, 4]>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>>
// Remove W
#conv_compilation1 = #iree_codegen.compilation_info<
    lowering_config = #iree_cpu.lowering_config<distribution = [0, 7, 7, 64, 0, 0, 0], vector_common_parallel = [1, 7, 1, 4, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 3, 1, 4]>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>>
func.func @conv() {
  %input = util.unfoldable_constant dense<1.0> : tensor<36x7x7x512xf32>
  %filter = util.unfoldable_constant dense<1.0> : tensor<3x3x512x512xf32>
  %0 = "stablehlo.convolution"(%input, %filter) {
    compilation_info = #conv_compilation0,
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>,
    feature_group_count = 1 : i64,
    padding = dense<1> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<36x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<36x7x7x512xf32>
  %1 = "stablehlo.convolution"(%input, %filter) {
    compilation_info = #conv_compilation1,
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>,
    feature_group_count = 1 : i64,
    padding = dense<1> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<36x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<36x7x7x512xf32>
  check.expect_almost_eq(%0, %1) : tensor<36x7x7x512xf32>
  return
}
