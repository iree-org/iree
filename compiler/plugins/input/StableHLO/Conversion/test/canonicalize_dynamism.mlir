// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.

// CHECK-LABEL: @dynamic_pad_constant
// CHECK: return %{{.+}} : tensor<6xf32>
func.func @dynamic_pad_constant(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<6xf32> {
  %low = stablehlo.constant dense<1> : tensor<1xi64>
  %high = stablehlo.constant dense<1> : tensor<1xi64>
  %interior = stablehlo.constant dense<0> : tensor<1xi64>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %low, %high, %interior)
    : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
    -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// -----

// CHECK-LABEL: @dynamic_gather_constant
// CHECK: return %{{.+}} : tensor<2x3x2x2xi32>
func.func @dynamic_gather_constant(%arg0: tensor<3x4x2xi32>,
                                   %arg1: tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32> {
  %sizes = stablehlo.constant dense<[1, 2, 2]> : tensor<3xi64>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %sizes) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3],
                                          collapsed_slice_dims = [0],
                                          start_index_map = [1, 0],
                                          index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi32>
  return %0 : tensor<2x3x2x2xi32>
}

// -----

// CHECK-LABEL: @dynamic_conv_constant
// CHECK: return %{{.+}} : tensor<1x6x6x1xf32>
func.func @dynamic_conv_constant(%arg0: tensor<1x8x8x1xf32>,
                                 %arg1: tensor<3x3x1x1xf32>) -> tensor<1x6x6x1xf32> {
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>
  } : (tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>, tensor<2x2xi64>) -> tensor<1x6x6x1xf32>
  return %0 : tensor<1x6x6x1xf32>
}
