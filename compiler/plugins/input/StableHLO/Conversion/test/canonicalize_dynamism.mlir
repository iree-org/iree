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

// Negative edges generate the final shape without materialising cropped padding.
// CHECK-LABEL: @dynamic_pad_negative_edges
// CHECK-DAG: %[[ONE_INDEX:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[ONE:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[THREE:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[PAD:.+]] = tensor.extract %arg1[] : tensor<f32>
// CHECK: %[[RESULT:.+]] = linalg.generic {{.*}} outs(%{{.+}} : tensor<1x5x4xf32>)
// CHECK: arith.cmpi ult, %{{.+}}, %[[THREE]] : i64
// CHECK: %[[COL:.+]] = linalg.index 2 : index
// CHECK: %[[COL64:.+]] = arith.index_castui %[[COL]] : index to i64
// CHECK: arith.cmpi sge, %[[COL64]], %[[ONE]] : i64
// CHECK: %[[OFFSET:.+]] = arith.subi %[[COL64]], %[[ONE]] : i64
// CHECK: %[[SOURCE_COL:.+]] = arith.index_castui %[[OFFSET]] : i64 to index
// CHECK: %[[VALUE:.+]] = scf.if %{{.+}} -> (f32) {
// CHECK-NEXT: %[[ELEMENT:.+]] = tensor.extract %arg0[%[[ONE_INDEX]], %{{.+}}, %[[SOURCE_COL]]]
// CHECK-NEXT: scf.yield %[[ELEMENT]] : f32
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[PAD]] : f32
// CHECK: linalg.yield %[[VALUE]] : f32
// CHECK: return %[[RESULT]] : tensor<1x5x4xf32>
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

// A crop with interior padding maps output coordinates through the stride.
// CHECK-LABEL: @dynamic_pad_negative_high_interior
// CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[ZERO:.+]] = arith.constant 0 : i64
// CHECK-DAG: %[[THREE:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[PAD:.+]] = tensor.extract %arg1[] : tensor<f32>
// CHECK: %[[RESULT:.+]] = linalg.generic {{.*}} outs(%{{.+}} : tensor<1x4x13xf32>)
// CHECK: %[[COL:.+]] = linalg.index 2 : index
// CHECK: %[[COL64:.+]] = arith.index_castui %[[COL]] : index to i64
// CHECK: %[[SOURCE_COL64:.+]] = arith.divui %[[COL64]], %[[THREE]] : i64
// CHECK: %[[REMAINDER:.+]] = arith.remui %[[COL64]], %[[THREE]] : i64
// CHECK: arith.cmpi eq, %[[REMAINDER]], %[[ZERO]] : i64
// CHECK: %[[SOURCE_COL:.+]] = arith.index_castui %[[SOURCE_COL64]] : i64 to index
// CHECK: %[[VALUE:.+]] = scf.if %{{.+}} -> (f32) {
// CHECK-NEXT: %[[ELEMENT:.+]] = tensor.extract %arg0[%[[ZERO_INDEX]], %{{.+}}, %[[SOURCE_COL]]]
// CHECK-NEXT: scf.yield %[[ELEMENT]] : f32
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[PAD]] : f32
// CHECK: linalg.yield %[[VALUE]] : f32
// CHECK: return %[[RESULT]] : tensor<1x4x13xf32>
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
// CHECK-DAG: %[[FOUR:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[FIVE:.+]] = arith.constant 5 : i64
// CHECK: linalg.generic
// CHECK: %[[CLAMP0:.+]] = arith.minsi %{{.+}}, %[[FOUR]] : i64
// CHECK: arith.index_cast %[[CLAMP0]] : i64 to index
// CHECK: %[[CLAMP1:.+]] = arith.minsi %{{.+}}, %[[FIVE]] : i64
// CHECK: arith.index_cast %[[CLAMP1]] : i64 to index
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
// CHECK-DAG: %[[NEGATIVE_ONE:.+]] = arith.constant -1 : i64
// CHECK-DAG: %[[PAD:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CROPPED:.+]] = linalg.generic {{.*}} outs(%{{.+}} : tensor<2x7x4x3xf32>)
// CHECK: %[[ROW:.+]] = linalg.index 1 : index
// CHECK: %[[ROW64:.+]] = arith.index_castui %[[ROW]] : index to i64
// CHECK: %[[SOURCE_ROW64:.+]] = arith.subi %[[ROW64]], %[[NEGATIVE_ONE]] : i64
// CHECK: %[[SOURCE_ROW:.+]] = arith.index_castui %[[SOURCE_ROW64]] : i64 to index
// CHECK: %[[VALUE:.+]] = scf.if %{{.+}} -> (f32) {
// CHECK-NEXT: %[[ELEMENT:.+]] = tensor.extract %arg0[%{{.+}}, %[[SOURCE_ROW]], %{{.+}}, %{{.+}}]
// CHECK-NEXT: scf.yield %[[ELEMENT]] : f32
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[PAD]] : f32
// CHECK: linalg.yield %[[VALUE]] : f32
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: ins(%[[CROPPED]], %arg1 : tensor<2x7x4x3xf32>, tensor<3x2x3x4xf32>)
// CHECK: return %[[CONV]] : tensor<2x5x3x4xf32>
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
