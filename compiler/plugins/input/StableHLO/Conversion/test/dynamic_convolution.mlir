// RUN: iree-opt --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @dynamic_conv_dynamic_spatial
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dynamic_conv_dynamic_spatial(%a: tensor<2x8x6x3xf32>, %k: tensor<3x2x3x4xf32>, %p: tensor<2x2xi64>) -> tensor<2x?x?x4xf32> {
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>, lhs_dilation = array<i64: 1, 1>, rhs_dilation = array<i64: 1, 1>
  } : (tensor<2x8x6x3xf32>, tensor<3x2x3x4xf32>, tensor<2x2xi64>) -> tensor<2x?x?x4xf32>
  return %r : tensor<2x?x?x4xf32>
}

// CHECK-LABEL: func.func @convolution_1d
// CHECK: tensor.dim
// CHECK: arith.divsi
// CHECK: arith.select
// CHECK: scf.if
// CHECK: linalg.conv_1d_nwc_wcf
func.func @convolution_1d(%a: tensor<?x?x?xf32>, %w: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %r = "stablehlo.convolution"(%a, %w) {
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 2>
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @convolution_2d
// CHECK: tensor.dim
// CHECK: arith.divsi
// CHECK: arith.select
// CHECK: scf.if
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @convolution_2d(%a: tensor<?x?x?x?xf32>, %w: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %r = "stablehlo.convolution"(%a, %w) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 2, 2>
  } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %r : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @convolution_3d
// CHECK: tensor.dim
// CHECK: arith.divsi
// CHECK: arith.select
// CHECK: scf.if
// CHECK: linalg.conv_3d_ndhwc_dhwcf
func.func @convolution_3d(%a: tensor<?x?x?x?x?xf32>, %w: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %r = "stablehlo.convolution"(%a, %w) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 2, 2, 2>
  } : (tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %r : tensor<?x?x?x?x?xf32>
}

// CHECK-LABEL: func.func @convolution_nchw
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @convolution_nchw(%a: tensor<1x3x?x?xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<1x4x?x?xf32> {
  %r = "stablehlo.convolution"(%a, %w) {
    dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>,
    lhs_dilation = array<i64: 1, 1>, rhs_dilation = array<i64: 1, 1>
  } : (tensor<1x3x?x?xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x?x?xf32>
  return %r : tensor<1x4x?x?xf32>
}

// CHECK-LABEL: func.func @feature_groups
// CHECK: tensor.expand_shape
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tensor.collapse_shape
func.func @feature_groups(%a: tensor<?x?x?xf32>, %w: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %r = "stablehlo.convolution"(%a, %w) {
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 2 : i64, batch_group_count = 1 : i64
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @batch_groups
// CHECK: tensor.expand_shape
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tensor.collapse_shape
func.func @batch_groups(%a: tensor<?x?x?xf32>, %w: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %r = "stablehlo.convolution"(%a, %w) {
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 1 : i64, batch_group_count = 2 : i64
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @depthwise
// CHECK: tensor.expand_shape
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tensor.collapse_shape
func.func @depthwise(%a: tensor<?x?x4xf32>, %w: tensor<3x1x8xf32>) -> tensor<?x?x8xf32> {
  %r = "stablehlo.convolution"(%a, %w) {
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 4 : i64, batch_group_count = 1 : i64
  } : (tensor<?x?x4xf32>, tensor<3x1x8xf32>) -> tensor<?x?x8xf32>
  return %r : tensor<?x?x8xf32>
}

// CHECK-LABEL: func.func @grouped_dynamic_batch
// CHECK: tensor.expand_shape
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tensor.collapse_shape
func.func @grouped_dynamic_batch(%a: tensor<?x5x4xf32>, %w: tensor<3x2x6xf32>) -> tensor<?x3x6xf32> {
  %r = "stablehlo.convolution"(%a, %w) {
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 2 : i64, batch_group_count = 1 : i64
  } : (tensor<?x5x4xf32>, tensor<3x2x6xf32>) -> tensor<?x3x6xf32>
  return %r : tensor<?x3x6xf32>
}
