// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.

// Every dim and every amount differs so a swapped or misplaced fold is caught.
// CHECK-LABEL: @dynamic_pad_constant
// CHECK: tensor.pad %arg0 low[1, 0, 2] high[0, 2, 1]
// CHECK: return %{{.+}} : tensor<3x5x8xf32>
func.func @dynamic_pad_constant(%arg0: tensor<2x3x5xf32>, %arg1: tensor<f32>) -> tensor<3x5x8xf32> {
  %low = stablehlo.constant dense<[1, 0, 2]> : tensor<3xi64>
  %high = stablehlo.constant dense<[0, 2, 1]> : tensor<3xi64>
  %interior = stablehlo.constant dense<0> : tensor<3xi64>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %low, %high, %interior)
    : (tensor<2x3x5xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>)
    -> tensor<3x5x8xf32>
  return %0 : tensor<3x5x8xf32>
}

// -----

// Interior padding becomes the stride of the insert.
// CHECK-LABEL: @dynamic_pad_interior
// CHECK: linalg.fill
// CHECK: tensor.insert_slice %arg0 into %{{.+}}[0, 1, 0] [2, 3, 5] [2, 1, 3]
// CHECK: return %{{.+}} : tensor<4x7x15xf32>
func.func @dynamic_pad_interior(%arg0: tensor<2x3x5xf32>, %arg1: tensor<f32>) -> tensor<4x7x15xf32> {
  %low = stablehlo.constant dense<[0, 1, 0]> : tensor<3xi64>
  %high = stablehlo.constant dense<[1, 3, 2]> : tensor<3xi64>
  %interior = stablehlo.constant dense<[1, 0, 2]> : tensor<3xi64>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %low, %high, %interior)
    : (tensor<2x3x5xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>)
    -> tensor<4x7x15xf32>
  return %0 : tensor<4x7x15xf32>
}

// -----

// Negative low and high amounts crop.
// CHECK-LABEL: @dynamic_pad_negative_edges
// CHECK: tensor.pad %arg0 low[0, 0, 1] high[0, 2, 2]
// CHECK: tensor.extract_slice %{{.+}}[1, 0, 0] [1, 5, 4] [1, 1, 1]
// CHECK: return %{{.+}} : tensor<1x5x4xf32>
func.func @dynamic_pad_negative_edges(%arg0: tensor<2x3x5xf32>, %arg1: tensor<f32>) -> tensor<1x5x4xf32> {
  %low = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi64>
  %high = stablehlo.constant dense<[0, 2, -2]> : tensor<3xi64>
  %interior = stablehlo.constant dense<0> : tensor<3xi64>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %low, %high, %interior)
    : (tensor<2x3x5xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>)
    -> tensor<1x5x4xf32>
  return %0 : tensor<1x5x4xf32>
}

// -----

// A negative high amount crops after the interior padding is applied.
// CHECK-LABEL: @dynamic_pad_negative_high_interior
// CHECK: tensor.insert_slice %arg0 into %{{.+}}[0, 0, 0] [2, 3, 5] [2, 1, 3]
// CHECK: tensor.extract_slice %{{.+}}[0, 0, 0] [1, 4, 13] [1, 1, 1]
// CHECK: return %{{.+}} : tensor<1x4x13xf32>
func.func @dynamic_pad_negative_high_interior(%arg0: tensor<2x3x5xf32>, %arg1: tensor<f32>) -> tensor<1x4x13xf32> {
  %low = stablehlo.constant dense<0> : tensor<3xi64>
  %high = stablehlo.constant dense<[-2, 1, 0]> : tensor<3xi64>
  %interior = stablehlo.constant dense<[1, 0, 2]> : tensor<3xi64>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %low, %high, %interior)
    : (tensor<2x3x5xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>)
    -> tensor<1x4x13xf32>
  return %0 : tensor<1x4x13xf32>
}

// -----

// Constant amounts on a dynamic operand fold to a pad with a dynamic result.
// CHECK-LABEL: @dynamic_pad_dynamic_operand
// CHECK: tensor.pad %arg0 low[1, 0] high[2, 4]
// CHECK: return %{{.+}} : tensor<?x7xf32>
func.func @dynamic_pad_dynamic_operand(%arg0: tensor<?x3xf32>, %arg1: tensor<f32>) -> tensor<?x7xf32> {
  %low = stablehlo.constant dense<[1, 0]> : tensor<2xi64>
  %high = stablehlo.constant dense<[2, 4]> : tensor<2xi64>
  %interior = stablehlo.constant dense<0> : tensor<2xi64>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %low, %high, %interior)
    : (tensor<?x3xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>)
    -> tensor<?x7xf32>
  return %0 : tensor<?x7xf32>
}

// -----

// The low amounts are a constant only after the concatenate folds.
// CHECK-LABEL: @dynamic_pad_folded_shape
// CHECK: tensor.pad %arg0 low[1, 2] high[0, 3]
// CHECK: return %{{.+}} : tensor<5x10xf32>
func.func @dynamic_pad_folded_shape(%arg0: tensor<4x5xf32>, %arg1: tensor<f32>) -> tensor<5x10xf32> {
  %c1 = stablehlo.constant dense<1> : tensor<1xi32>
  %c2 = stablehlo.constant dense<2> : tensor<1xi32>
  %low = stablehlo.concatenate %c1, %c2, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %high = stablehlo.constant dense<[0, 3]> : tensor<2xi32>
  %interior = stablehlo.constant dense<0> : tensor<2xi32>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %low, %high, %interior)
    : (tensor<4x5xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
    -> tensor<5x10xf32>
  return %0 : tensor<5x10xf32>
}

// -----

// The amounts are constants only after @low and @high are inlined.
// CHECK-LABEL: @dynamic_pad_after_inline
// CHECK: tensor.pad %arg0 low[1, 0] high[0, 3]
// CHECK: return %{{.+}} : tensor<3x8xf32>
func.func private @low() -> tensor<2xi64> {
  %c = stablehlo.constant dense<[1, 0]> : tensor<2xi64>
  return %c : tensor<2xi64>
}
func.func private @high() -> tensor<2xi64> {
  %c = stablehlo.constant dense<[0, 3]> : tensor<2xi64>
  return %c : tensor<2xi64>
}
func.func @dynamic_pad_after_inline(%arg0: tensor<2x5xf32>, %arg1: tensor<f32>) -> tensor<3x8xf32> {
  %low = call @low() : () -> tensor<2xi64>
  %high = call @high() : () -> tensor<2xi64>
  %interior = stablehlo.constant dense<0> : tensor<2xi64>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %low, %high, %interior)
    : (tensor<2x5xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>)
    -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
}

// -----

// Start indices are clamped to operand_dim - slice_size: 5 - 1 and 8 - 3.
// CHECK-LABEL: @dynamic_gather_constant
// CHECK-DAG: arith.constant 4 : index
// CHECK-DAG: arith.constant 5 : index
// CHECK: linalg.generic
// CHECK: return %{{.+}} : tensor<2x7x3x2xi32>
func.func @dynamic_gather_constant(%arg0: tensor<5x8x2xi32>,
                                   %arg1: tensor<2x7x2xi64>) -> tensor<2x7x3x2xi32> {
  %sizes = stablehlo.constant dense<[1, 3, 2]> : tensor<3xi32>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %sizes) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3],
                                          collapsed_slice_dims = [0],
                                          start_index_map = [1, 0],
                                          index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<5x8x2xi32>, tensor<2x7x2xi64>, tensor<3xi32>) -> tensor<2x7x3x2xi32>
  return %0 : tensor<2x7x3x2xi32>
}

// -----

// A zero slice size gives an empty result.
// CHECK-LABEL: @dynamic_gather_zero_slice
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<6x0x2xi32>
// CHECK: return %[[EMPTY]]
func.func @dynamic_gather_zero_slice(%arg0: tensor<5x8x2xi32>,
                                     %arg1: tensor<6x2xi64>) -> tensor<6x0x2xi32> {
  %sizes = stablehlo.constant dense<[1, 0, 2]> : tensor<3xi64>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %sizes) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1, 2],
                                          collapsed_slice_dims = [0],
                                          start_index_map = [0, 1],
                                          index_vector_dim = 1>,
    indices_are_sorted = false
  } : (tensor<5x8x2xi32>, tensor<6x2xi64>, tensor<3xi64>) -> tensor<6x0x2xi32>
  return %0 : tensor<6x0x2xi32>
}

// -----

// Row i of the padding is [low, high] of spatial dim i.
// CHECK-LABEL: @dynamic_conv_constant
// CHECK: tensor.pad %arg0 low[0, 1, 0, 0] high[0, 2, 1, 0]
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<2x11x7x3xf32>, tensor<3x2x3x4xf32>)
// CHECK: return %{{.+}} : tensor<2x9x6x4xf32>
func.func @dynamic_conv_constant(%arg0: tensor<2x8x6x3xf32>,
                                 %arg1: tensor<3x2x3x4xf32>) -> tensor<2x9x6x4xf32> {
  %padding = stablehlo.constant dense<[[1, 2], [0, 1]]> : tensor<2x2xi32>
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>
  } : (tensor<2x8x6x3xf32>, tensor<3x2x3x4xf32>, tensor<2x2xi32>) -> tensor<2x9x6x4xf32>
  return %0 : tensor<2x9x6x4xf32>
}

// -----

// Negative padding crops the input before the convolution.
// CHECK-LABEL: @dynamic_conv_negative_padding
// CHECK: tensor.extract_slice %{{.+}}[0, 1, 0, 0] [2, 7, 4, 3] [1, 1, 1, 1]
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<2x7x4x3xf32>, tensor<3x2x3x4xf32>)
// CHECK: return %{{.+}} : tensor<2x5x3x4xf32>
func.func @dynamic_conv_negative_padding(%arg0: tensor<2x8x6x3xf32>,
                                         %arg1: tensor<3x2x3x4xf32>) -> tensor<2x5x3x4xf32> {
  %padding = stablehlo.constant dense<[[-1, 0], [0, -2]]> : tensor<2x2xi64>
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>
  } : (tensor<2x8x6x3xf32>, tensor<3x2x3x4xf32>, tensor<2x2xi64>) -> tensor<2x5x3x4xf32>
  return %0 : tensor<2x5x3x4xf32>
}
