// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.

// CHECK-LABEL: @dynamic_reshape
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x2xf32>, %[[SHAPE:.+]]: tensor<1xi64>) -> tensor<?xf32>
// CHECK: %[[R:.+]] = tensor.reshape %[[ARG0]](%[[SHAPE]]) : (tensor<4x2xf32>, tensor<1xi64>) -> tensor<?xf32>
// CHECK: return %[[R]] : tensor<?xf32>
func.func @dynamic_reshape(%a: tensor<4x2xf32>, %s: tensor<1xi64>) -> tensor<?xf32> {
  %r = "stablehlo.dynamic_reshape"(%a, %s) : (tensor<4x2xf32>, tensor<1xi64>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: @dynamic_reshape_rank3
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x3x5xf32>, %[[SHAPE:.+]]: tensor<2xi64>)
// CHECK: tensor.reshape %[[ARG0]](%[[SHAPE]]) : (tensor<2x3x5xf32>, tensor<2xi64>) -> tensor<?x?xf32>
func.func @dynamic_reshape_rank3(%a: tensor<2x3x5xf32>, %s: tensor<2xi64>) -> tensor<?x?xf32> {
  %r = "stablehlo.dynamic_reshape"(%a, %s) : (tensor<2x3x5xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// A dynamic operand and an i32 shape.
// CHECK-LABEL: @dynamic_reshape_dynamic_operand
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x3xf32>, %[[SHAPE:.+]]: tensor<3xi32>)
// CHECK: tensor.reshape %[[ARG0]](%[[SHAPE]]) : (tensor<?x3xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
func.func @dynamic_reshape_dynamic_operand(%a: tensor<?x3xf32>, %s: tensor<3xi32>) -> tensor<?x?x?xf32> {
  %r = "stablehlo.dynamic_reshape"(%a, %s) : (tensor<?x3xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// One result dim is static.
// CHECK-LABEL: @dynamic_reshape_mixed_result
// CHECK: tensor.reshape %{{.+}}(%{{.+}}) : (tensor<3x5x?xf32>, tensor<2xi64>) -> tensor<?x15xf32>
func.func @dynamic_reshape_mixed_result(%a: tensor<3x5x?xf32>, %s: tensor<2xi64>) -> tensor<?x15xf32> {
  %r = "stablehlo.dynamic_reshape"(%a, %s) : (tensor<3x5x?xf32>, tensor<2xi64>) -> tensor<?x15xf32>
  return %r : tensor<?x15xf32>
}

// -----

// CHECK-LABEL: @dynamic_conv_lhs_dilation
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK: return %{{.+}} : tensor<1x5x5x1xf32>
func.func @dynamic_conv_lhs_dilation(%a: tensor<1x4x4x1xf32>, %k: tensor<3x3x1x1xf32>, %p: tensor<2x2xi64>) -> tensor<1x5x5x1xf32> {
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>, lhs_dilation = array<i64: 2, 2>, rhs_dilation = array<i64: 1, 1>
  } : (tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>, tensor<2x2xi64>) -> tensor<1x5x5x1xf32>
  return %r : tensor<1x5x5x1xf32>
}

// -----

// The padding amounts are runtime values.
// CHECK-LABEL: @dynamic_pad
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>, %[[VAL:.+]]: tensor<f32>, %[[LO:.+]]: tensor<1xi64>, %[[HI:.+]]: tensor<1xi64>, %[[IN:.+]]: tensor<1xi64>)
// CHECK-DAG: %[[LOWX:.+]] = tensor.extract %[[LO]]
// CHECK-DAG: %[[LOWV:.+]] = arith.index_cast %[[LOWX]]
// CHECK-DAG: %[[HIGHX:.+]] = tensor.extract %[[HI]]
// CHECK-DAG: %[[HIGHV:.+]] = arith.index_cast %[[HIGHX]]
// CHECK-DAG: %[[INX:.+]] = tensor.extract %[[IN]]
// CHECK-DAG: %[[INV:.+]] = arith.index_cast %[[INX]]
// CHECK-DAG: %[[LOWPOS:.+]] = arith.maxsi %[[LOWV]], %c0
// CHECK-DAG: %[[HIGHPOS:.+]] = arith.maxsi %[[HIGHV]], %c0
// CHECK-DAG: %[[LOWNEG:.+]] = arith.maxsi %{{.+}}, %c0
// CHECK-DAG: %[[FILLDIM:.+]] = arith.addi %{{.+}}, %[[HIGHPOS]]
// CHECK-DAG: %[[RESULTDIM:.+]] = arith.addi %{{.+}}, %[[HIGHV]]
// CHECK-DAG: %[[STRIDE:.+]] = arith.addi %[[INV]], %c1
// CHECK: %[[IF:.+]] = scf.if
// CHECK: %[[EMPTY:.+]] = tensor.empty(%[[FILLDIM]]) : tensor<?xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[EMPTY]] : tensor<?xf32>)
// CHECK: %[[INS:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][%[[LOWPOS]]] [4] [%[[STRIDE]]] : tensor<4xf32> into tensor<?xf32>
// CHECK: %[[R:.+]] = tensor.extract_slice %[[INS]][%[[LOWNEG]]] [%[[RESULTDIM]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK: scf.yield %[[R]]
// CHECK: } else {
// CHECK: %[[ZERO_SIZE:.+]] = tensor.empty
// CHECK: scf.yield %[[ZERO_SIZE]]
// CHECK: %[[TIED:.+]] = flow.tensor.tie_shape %[[IF]]
// CHECK: return %[[TIED]]
func.func @dynamic_pad(%a: tensor<4xf32>, %v: tensor<f32>, %lo: tensor<1xi64>, %hi: tensor<1xi64>, %in: tensor<1xi64>) -> tensor<?xf32> {
  %r = "stablehlo.dynamic_pad"(%a, %v, %lo, %hi, %in) : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// The padding amounts are runtime values but the result shape is static.
// CHECK-LABEL: @dynamic_pad_static_result
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x3xf32>, %[[VAL:.+]]: tensor<f32>, %[[LO:.+]]: tensor<2xi64>, %[[HI:.+]]: tensor<2xi64>, %[[IN:.+]]: tensor<2xi64>) -> tensor<5x9xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[EMPTY]] : tensor<?x?xf32>)
// CHECK: %[[INS:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][%{{.+}}, %{{.+}}] [2, 3] [%{{.+}}, %{{.+}}] : tensor<2x3xf32> into tensor<?x?xf32>
// CHECK: %[[R:.+]] = tensor.extract_slice %[[INS]][%{{.+}}, %{{.+}}] [5, 9] [1, 1] : tensor<?x?xf32> to tensor<5x9xf32>
// CHECK: return %[[R]] : tensor<5x9xf32>
func.func @dynamic_pad_static_result(%a: tensor<2x3xf32>, %v: tensor<f32>, %lo: tensor<2xi64>, %hi: tensor<2xi64>, %in: tensor<2xi64>) -> tensor<5x9xf32> {
  %r = "stablehlo.dynamic_pad"(%a, %v, %lo, %hi, %in) : (tensor<2x3xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x9xf32>
  return %r : tensor<5x9xf32>
}

// -----

// Rank 3, every dim distinct: the insert keeps the operand sizes and takes
// one runtime offset and one runtime stride per dim.
// CHECK-LABEL: @dynamic_pad_rank3
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x3x5xf32>,
// CHECK: %[[IF:.+]] = scf.if
// CHECK: %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}, %{{.+}}) : tensor<?x?x?xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[EMPTY]] : tensor<?x?x?xf32>)
// CHECK: %[[INS:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][%{{.+}}, %{{.+}}, %{{.+}}] [2, 3, 5] [%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x3x5xf32> into tensor<?x?x?xf32>
// CHECK: %[[R:.+]] = tensor.extract_slice %[[INS]][%{{.+}}, %{{.+}}, %{{.+}}] [%{{.+}}, %{{.+}}, %{{.+}}] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
// CHECK: scf.yield %[[R]]
// CHECK: } else {
// CHECK: %[[ZERO_SIZE:.+]] = tensor.empty
// CHECK: scf.yield %[[ZERO_SIZE]]
// CHECK: %[[TIED:.+]] = flow.tensor.tie_shape %[[IF]]
// CHECK: return %[[TIED]]
func.func @dynamic_pad_rank3(%a: tensor<2x3x5xf32>, %v: tensor<f32>, %lo: tensor<3xi64>, %hi: tensor<3xi64>, %in: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %r = "stablehlo.dynamic_pad"(%a, %v, %lo, %hi, %in) : (tensor<2x3x5xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// A dynamic operand: the insert sizes come from tensor.dim.
// CHECK-LABEL: @dynamic_pad_dynamic_operand
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x3x?xf32>,
// CHECK-DAG: tensor.dim %[[ARG0]], %c0
// CHECK-DAG: tensor.dim %[[ARG0]], %c2
// CHECK: tensor.insert_slice %[[ARG0]] into %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] [%{{.+}}, 3, %{{.+}}] [%{{.+}}, %{{.+}}, %{{.+}}] : tensor<?x3x?xf32> into tensor<?x?x?xf32>
func.func @dynamic_pad_dynamic_operand(%a: tensor<?x3x?xf32>, %v: tensor<f32>, %lo: tensor<3xi64>, %hi: tensor<3xi64>, %in: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %r = "stablehlo.dynamic_pad"(%a, %v, %lo, %hi, %in) : (tensor<?x3x?xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// Constant zero interior with runtime edges: the fold needs all three
// constant, so this pattern runs and the strides fold to 1.
// CHECK-LABEL: @dynamic_pad_constant_interior
// CHECK: tensor.insert_slice %{{.+}} into %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] [2, 3, 5] [1, 1, 1]
func.func @dynamic_pad_constant_interior(%a: tensor<2x3x5xf32>, %v: tensor<f32>, %lo: tensor<3xi64>, %hi: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %in = stablehlo.constant dense<0> : tensor<3xi64>
  %r = "stablehlo.dynamic_pad"(%a, %v, %lo, %hi, %in) : (tensor<2x3x5xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// i32 amounts.
// CHECK-LABEL: @dynamic_pad_i32_amounts
// CHECK: arith.index_cast %{{.+}} : i32 to index
// CHECK: tensor.insert_slice %{{.+}} into %{{.+}}[%{{.+}}, %{{.+}}] [4, 7] [%{{.+}}, %{{.+}}] : tensor<4x7xf32> into tensor<?x?xf32>
func.func @dynamic_pad_i32_amounts(%a: tensor<4x7xf32>, %v: tensor<f32>, %lo: tensor<2xi32>, %hi: tensor<2xi32>, %in: tensor<2xi32>) -> tensor<?x?xf32> {
  %r = "stablehlo.dynamic_pad"(%a, %v, %lo, %hi, %in) : (tensor<4x7xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_conv
// CHECK: tensor.insert_slice
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK: return %{{.+}} : tensor<1x8x8x1xf32>
func.func @dynamic_conv(%a: tensor<1x8x8x1xf32>, %k: tensor<3x3x1x1xf32>, %p: tensor<2x2xi64>) -> tensor<1x8x8x1xf32> {
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>, lhs_dilation = array<i64: 1, 1>, rhs_dilation = array<i64: 1, 1>
  } : (tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>, tensor<2x2xi64>) -> tensor<1x8x8x1xf32>
  return %r : tensor<1x8x8x1xf32>
}

// -----

// NHWC with distinct dimensions: batch and feature sizes remain static.
// CHECK-LABEL: @dynamic_conv_nhwc
// CHECK: %[[INS:.+]] = tensor.insert_slice %{{.+}} into %{{.+}}[0, %{{.+}}, %{{.+}}, 0] [2, 8, 6, 3] [1, 1, 1, 1] : tensor<2x8x6x3xf32> into tensor<2x?x?x3xf32>
// CHECK: tensor.extract_slice %[[INS]][0, %{{.+}}, %{{.+}}, 0] [2, %{{.+}}, %{{.+}}, 3] [1, 1, 1, 1] : tensor<2x?x?x3xf32> to tensor<2x?x?x3xf32>
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<2x?x?x3xf32>, tensor<3x2x3x4xf32>)
// CHECK: return %{{.+}} : tensor<2x8x6x4xf32>
func.func @dynamic_conv_nhwc(%a: tensor<2x8x6x3xf32>, %k: tensor<3x2x3x4xf32>, %p: tensor<2x2xi64>) -> tensor<2x8x6x4xf32> {
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>, lhs_dilation = array<i64: 1, 1>, rhs_dilation = array<i64: 1, 1>
  } : (tensor<2x8x6x3xf32>, tensor<3x2x3x4xf32>, tensor<2x2xi64>) -> tensor<2x8x6x4xf32>
  return %r : tensor<2x8x6x4xf32>
}

// -----

// lhs_dilation [2, 3] becomes the insert strides [1, 2, 3, 1].
// CHECK-LABEL: @dynamic_conv_lhs_dilation_nhwc
// CHECK: tensor.insert_slice %{{.+}} into %{{.+}}[0, %{{.+}}, %{{.+}}, 0] [2, 4, 3, 3] [1, 2, 3, 1]
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: dilations = dense<1>
// CHECK: return %{{.+}} : tensor<2x5x6x5xf32>
func.func @dynamic_conv_lhs_dilation_nhwc(%a: tensor<2x4x3x3xf32>, %k: tensor<3x2x3x5xf32>, %p: tensor<2x2xi64>) -> tensor<2x5x6x5xf32> {
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>, lhs_dilation = array<i64: 2, 3>, rhs_dilation = array<i64: 1, 1>
  } : (tensor<2x4x3x3xf32>, tensor<3x2x3x5xf32>, tensor<2x2xi64>) -> tensor<2x5x6x5xf32>
  return %r : tensor<2x5x6x5xf32>
}

// -----

// CHECK-LABEL: @pad_static_operand_dynamic_result
// CHECK: tensor.pad %{{.+}} low[1] high[1]
// CHECK: tensor<4xf32> to tensor<6xf32>
// CHECK: tensor.cast %{{.+}} : tensor<6xf32> to tensor<?xf32>
func.func @pad_static_operand_dynamic_result(%input: tensor<4xf32>, %value: tensor<f32>) -> tensor<?xf32> {
  %r = stablehlo.pad %input, %value, low = [1], high = [1], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: @pad_partially_static_operand
// CHECK: tensor.pad %{{.+}} low[1, 2] high[1, 2]
// CHECK: tensor<4x?xf32> to tensor<6x?xf32>
// CHECK: tensor.cast %{{.+}} : tensor<6x?xf32> to tensor<?x?xf32>
func.func @pad_partially_static_operand(%input: tensor<4x?xf32>, %value: tensor<f32>) -> tensor<?x?xf32> {
  %r = stablehlo.pad %input, %value, low = [1, 2], high = [1, 2], interior = [0, 0] : (tensor<4x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// Canonicalizing constant padding must preserve a valid lowering.
// CHECK-LABEL: @dynamic_pad_constant_padding_dynamic_result
// CHECK: tensor.pad
// CHECK: tensor<4xf32> to tensor<6xf32>
// CHECK: tensor.cast %{{.+}} : tensor<6xf32> to tensor<?xf32>
func.func @dynamic_pad_constant_padding_dynamic_result(%input: tensor<4xf32>, %value: tensor<f32>) -> tensor<?xf32> {
  %one = stablehlo.constant dense<1> : tensor<1xi64>
  %zero = stablehlo.constant dense<0> : tensor<1xi64>
  %r = stablehlo.dynamic_pad %input, %value, %one, %one, %zero : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: @pad_partially_static_operand_distinct
// CHECK: tensor.pad %{{.+}} low[1, 3] high[2, 5]
// CHECK: tensor<4x?xf32> to tensor<7x?xf32>
// CHECK: tensor.cast %{{.+}} : tensor<7x?xf32> to tensor<?x?xf32>
func.func @pad_partially_static_operand_distinct(%input: tensor<4x?xf32>, %value: tensor<f32>) -> tensor<?x?xf32> {
  %r = stablehlo.pad %input, %value, low = [1, 3], high = [2, 5], interior = [0, 0] : (tensor<4x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_pad_constant_padding_rank2
// CHECK: tensor.pad %{{.+}} low[1, 3] high[2, 5]
// CHECK: tensor<4x7xf32> to tensor<7x15xf32>
// CHECK: tensor.cast %{{.+}} : tensor<7x15xf32> to tensor<?x?xf32>
func.func @dynamic_pad_constant_padding_rank2(%input: tensor<4x7xf32>, %value: tensor<f32>) -> tensor<?x?xf32> {
  %low = stablehlo.constant dense<[1, 3]> : tensor<2xi64>
  %high = stablehlo.constant dense<[2, 5]> : tensor<2xi64>
  %interior = stablehlo.constant dense<0> : tensor<2xi64>
  %r = stablehlo.dynamic_pad %input, %value, %low, %high, %interior : (tensor<4x7xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// A runtime-shaped pad with negative edges and interior padding pads
// nonnegatively into scratch, then crops the negative edges.
// CHECK-LABEL: @pad_dynamic_negative_interior
// CHECK-SAME: (%[[ARG:[a-zA-Z0-9]+]]: tensor<?x5xf32>
// CHECK: %[[FILL:.+]] = linalg.fill {{.+}} -> tensor<?x11xf32>
// CHECK: %[[INSERTED:.+]] = tensor.insert_slice %[[ARG]] into %[[FILL]][0, 2] [%{{.+}}, 5] [2, 2] : tensor<?x5xf32> into tensor<?x11xf32>
// CHECK: tensor.extract_slice %[[INSERTED]][1, 0] [%{{.+}}, 10] [1, 1] : tensor<?x11xf32> to tensor<?x10xf32>
func.func @pad_dynamic_negative_interior(%arg0: tensor<?x5xf32>, %pad: tensor<f32>) -> tensor<?x10xf32> {
  %0 = stablehlo.pad %arg0, %pad, low = [-1, 2], high = [3, -1], interior = [1, 1] : (tensor<?x5xf32>, tensor<f32>) -> tensor<?x10xf32>
  return %0 : tensor<?x10xf32>
}

// -----

// The runtime operand size decides whether this dimension expands.
// CHECK-LABEL: @dynamic_broadcast_undecidable
// CHECK: %[[EXPANDS:.+]] = arith.cmpi eq
// CHECK: linalg.generic
// CHECK-NOT: arith.cmpi
// CHECK: arith.select %[[EXPANDS]]
// CHECK: tensor.extract %{{.+}} : tensor<?xf32>
// CHECK: return %{{.+}} : tensor<?x?xf32>
func.func @dynamic_broadcast_undecidable(%a: tensor<?xf32>, %s: tensor<2xi64>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}


// -----

// Two unknown dimensions use runtime expansion checks; the static 1 expands
// directly. A permuted output mapping also exercises nonidentity dimensions.
// CHECK-LABEL: @dynamic_broadcast_mixed_permuted
// CHECK-COUNT-2: arith.cmpi eq
// CHECK: linalg.generic
// CHECK-NOT: arith.cmpi
// CHECK-COUNT-2: arith.select
// CHECK-NOT: arith.select
// CHECK: tensor.extract %{{.+}} : tensor<?x1x?xf32>
// CHECK: return %{{.+}} : tensor<?x?x?x?xf32>
func.func @dynamic_broadcast_mixed_permuted(%a: tensor<?x1x?xf32>, %s: tensor<4xi64>) -> tensor<?x?x?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [3, 1, 0] : (tensor<?x1x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %r : tensor<?x?x?x?xf32>
}


// -----

// A partial annotation leaves one dynamic dimension for the runtime path.
// CHECK-LABEL: @dynamic_broadcast_partial_annotation
// CHECK: %[[EXPANDS:.+]] = arith.cmpi eq
// CHECK-NOT: arith.cmpi
// CHECK: linalg.generic
// CHECK: arith.select %[[EXPANDS]]
// CHECK-NOT: arith.select
// CHECK: return %{{.+}} : tensor<?x?xf32>
func.func @dynamic_broadcast_partial_annotation(%a: tensor<?x?xf32>, %s: tensor<2xi64>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [0, 1] {known_nonexpanding_dimensions = array<i64: 0>} : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}


// -----

// An i8 truncation of n says nothing about n: n = 257 ties the operand to one
// element, which broadcasts to 257. The runtime check stays.
// CHECK-LABEL: @dynamic_broadcast_truncated_size
// CHECK: %[[EXPANDS:.+]] = arith.cmpi eq
// CHECK: linalg.generic
// CHECK: arith.select %[[EXPANDS]]
func.func @dynamic_broadcast_truncated_size(%a: tensor<?x3xf32>, %n: index) -> tensor<?x3xf32> {
  %small = arith.index_cast %n : index to i8
  %size = arith.index_cast %small : i8 to index
  %tied = flow.tensor.tie_shape %a : tensor<?x3xf32>{%size}
  %full = arith.index_cast %n : index to i64
  %three = arith.constant 3 : i64
  %shape = tensor.from_elements %full, %three : tensor<2xi64>
  %r = stablehlo.dynamic_broadcast_in_dim %tied, %shape, dims = [0, 1] : (tensor<?x3xf32>, tensor<2xi64>) -> tensor<?x3xf32>
  return %r : tensor<?x3xf32>
}


// -----

// Thirty squarings of n form a small graph whose product has 2^30 factors. The
// proof gives up within its budget and the runtime check stays.
// CHECK-LABEL: @dynamic_broadcast_squaring_chain
// CHECK: %[[EXPANDS:.+]] = arith.cmpi eq
// CHECK: linalg.generic
// CHECK: arith.select %[[EXPANDS]]
func.func @dynamic_broadcast_squaring_chain(%a: tensor<?x2xf32>, %n: tensor<i32>) -> tensor<?x2xf32> {
  %p0 = stablehlo.multiply %n, %n : tensor<i32>
  %p1 = stablehlo.multiply %p0, %p0 : tensor<i32>
  %p2 = stablehlo.multiply %p1, %p1 : tensor<i32>
  %p3 = stablehlo.multiply %p2, %p2 : tensor<i32>
  %p4 = stablehlo.multiply %p3, %p3 : tensor<i32>
  %p5 = stablehlo.multiply %p4, %p4 : tensor<i32>
  %p6 = stablehlo.multiply %p5, %p5 : tensor<i32>
  %p7 = stablehlo.multiply %p6, %p6 : tensor<i32>
  %p8 = stablehlo.multiply %p7, %p7 : tensor<i32>
  %p9 = stablehlo.multiply %p8, %p8 : tensor<i32>
  %p10 = stablehlo.multiply %p9, %p9 : tensor<i32>
  %p11 = stablehlo.multiply %p10, %p10 : tensor<i32>
  %p12 = stablehlo.multiply %p11, %p11 : tensor<i32>
  %p13 = stablehlo.multiply %p12, %p12 : tensor<i32>
  %p14 = stablehlo.multiply %p13, %p13 : tensor<i32>
  %p15 = stablehlo.multiply %p14, %p14 : tensor<i32>
  %p16 = stablehlo.multiply %p15, %p15 : tensor<i32>
  %p17 = stablehlo.multiply %p16, %p16 : tensor<i32>
  %p18 = stablehlo.multiply %p17, %p17 : tensor<i32>
  %p19 = stablehlo.multiply %p18, %p18 : tensor<i32>
  %p20 = stablehlo.multiply %p19, %p19 : tensor<i32>
  %p21 = stablehlo.multiply %p20, %p20 : tensor<i32>
  %p22 = stablehlo.multiply %p21, %p21 : tensor<i32>
  %p23 = stablehlo.multiply %p22, %p22 : tensor<i32>
  %p24 = stablehlo.multiply %p23, %p23 : tensor<i32>
  %p25 = stablehlo.multiply %p24, %p24 : tensor<i32>
  %p26 = stablehlo.multiply %p25, %p25 : tensor<i32>
  %p27 = stablehlo.multiply %p26, %p26 : tensor<i32>
  %p28 = stablehlo.multiply %p27, %p27 : tensor<i32>
  %p29 = stablehlo.multiply %p28, %p28 : tensor<i32>
  %size = stablehlo.reshape %p29 : (tensor<i32>) -> tensor<1xi32>
  %two = stablehlo.constant dense<2> : tensor<1xi32>
  %shape = stablehlo.concatenate %size, %two, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %r = stablehlo.dynamic_broadcast_in_dim %a, %shape, dims = [0, 1] : (tensor<?x2xf32>, tensor<2xi32>) -> tensor<?x2xf32>
  return %r : tensor<?x2xf32>
}


// -----

// A fully known expansion stays on the upstream affine-indexing path.
// CHECK: #map = affine_map<(d0, d1) -> (0)>
// CHECK-LABEL: @dynamic_broadcast_known_expanding
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0
// CHECK-NOT: tensor.extract
// CHECK-NOT: arith.select
// CHECK: return %{{.+}} : tensor<?x?xf32>
func.func @dynamic_broadcast_known_expanding(%a: tensor<?xf32>, %s: tensor<2xi64>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [1] {known_expanding_dimensions = array<i64: 0>} : (tensor<?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}


// -----

// An expansion the size proof settles for every dim also takes the affine
// path.
// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-LABEL: @dynamic_broadcast_inferred_nonexpanding
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MAP]],
// CHECK-SAME: ins(%arg0
// CHECK-NOT: tensor.extract
// CHECK-NOT: arith.select
// CHECK: return %{{.+}} : tensor<?x?x?xf32>
func.func @dynamic_broadcast_inferred_nonexpanding(%a: tensor<?x?xf32>, %n: index) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %a, %c1 : tensor<?x?xf32>
  %n64 = arith.index_cast %n : index to i64
  %d064 = arith.index_cast %d0 : index to i64
  %d164 = arith.index_cast %d1 : index to i64
  %shape = tensor.from_elements %n64, %d064, %d164 : tensor<3xi64>
  %r = stablehlo.dynamic_broadcast_in_dim %a, %shape, dims = [1, 2] : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}


// -----

// Without an upper bound on s, s + 1 truncated to i32 may wrap to a size of
// one, which broadcasts. The runtime check stays.
// CHECK-LABEL: @dynamic_broadcast_unbounded_sum
// CHECK: %[[EXPANDS:.+]] = arith.cmpi eq
// CHECK: linalg.generic
// CHECK: arith.select %[[EXPANDS]]
func.func @dynamic_broadcast_unbounded_sum(%cache: tensor<2x?x16xf32>, %row: tensor<2x1x16xf32>, %s: i64) -> tensor<2x?x16xf32> {
  %length = util.assume.int %s<umin = 1> : i64
  %rows = arith.index_cast %length : i64 to index
  %tied = flow.tensor.tie_shape %cache : tensor<2x?x16xf32>{%rows}
  %grown = stablehlo.concatenate %tied, %row, dim = 1 : (tensor<2x?x16xf32>, tensor<2x1x16xf32>) -> tensor<2x?x16xf32>
  %one = arith.constant 1 : i64
  %sum = arith.addi %length, %one : i64
  %size = arith.trunci %sum : i64 to i32
  %c2 = arith.constant 2 : i32
  %c16 = arith.constant 16 : i32
  %shape = tensor.from_elements %c2, %size, %c16 : tensor<3xi32>
  %r = stablehlo.dynamic_broadcast_in_dim %grown, %shape, dims = [0, 1, 2] : (tensor<2x?x16xf32>, tensor<3xi32>) -> tensor<2x?x16xf32>
  return %r : tensor<2x?x16xf32>
}


// -----

// An i32 product of x and y may wrap: 641 * 6700417 = 2^32 + 1 gives one row,
// which broadcasts to the exact product. The runtime check stays.
// CHECK-LABEL: @dynamic_broadcast_wrapped_product
// CHECK: %[[EXPANDS:.+]] = arith.cmpi eq
// CHECK: linalg.generic
// CHECK: arith.select %[[EXPANDS]]
func.func @dynamic_broadcast_wrapped_product(%a: tensor<?x3xf32>, %x: i32, %y: i32) -> tensor<?x3xf32> {
  %narrow = arith.muli %x, %y : i32
  %rows = arith.index_cast %narrow : i32 to index
  %tied = flow.tensor.tie_shape %a : tensor<?x3xf32>{%rows}
  %x64 = arith.extsi %x : i32 to i64
  %y64 = arith.extsi %y : i32 to i64
  %wide = arith.muli %x64, %y64 : i64
  %three = arith.constant 3 : i64
  %shape = tensor.from_elements %wide, %three : tensor<2xi64>
  %r = stablehlo.dynamic_broadcast_in_dim %tied, %shape, dims = [0, 1] : (tensor<?x3xf32>, tensor<2xi64>) -> tensor<?x3xf32>
  return %r : tensor<?x3xf32>
}


// -----

// Runtime slice sizes determine the offset dimensions of the result.
// CHECK-LABEL: @dynamic_gather_dynamic_result
// CHECK: tensor.empty(%{{.+}}, %{{.+}}) : tensor<2x3x?x?xi32>
// CHECK: arith.subi
// CHECK: linalg.generic
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: tensor.extract %{{.+}} : tensor<3x4x2xi32>
// CHECK: return %{{.+}} : tensor<2x3x?x?xi32>
func.func @dynamic_gather_dynamic_result(%a: tensor<3x4x2xi32>, %i: tensor<2x3x2xi64>, %s: tensor<3xi64>) -> tensor<2x3x?x?xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x?x?xi32>
  return %r : tensor<2x3x?x?xi32>
}


// -----

// Batching uses the result batch index, while the index vector follows the
// start-indices batch dimensions.
// CHECK-LABEL: @dynamic_gather_batching
// CHECK: linalg.generic
// CHECK: tensor.extract %{{.+}} : tensor<3x7x1xi64>
// CHECK: tensor.extract %{{.+}} : tensor<3x8x5xi32>
// CHECK: return %{{.+}} : tensor<3x7x5xi32>
func.func @dynamic_gather_batching(%a: tensor<3x8x5xi32>, %i: tensor<3x7x1xi64>, %s: tensor<3xi64>) -> tensor<3x7x5xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<3x8x5xi32>, tensor<3x7x1xi64>, tensor<3xi64>) -> tensor<3x7x5xi32>
  return %r : tensor<3x7x5xi32>
}


// -----

// index_vector_dim=0 and batching dim 1 use batch index 0 after removing the
// leading index-vector dimension.
// CHECK-LABEL: @dynamic_gather_leading_index_vector
// CHECK: linalg.generic
// CHECK: tensor.extract %{{.+}} : tensor<1x3x2xi64>
// CHECK: return %{{.+}} : tensor<3x2xi32>
func.func @dynamic_gather_leading_index_vector(%a: tensor<3x8xi32>, %i: tensor<1x3x2xi64>, %s: tensor<2xi64>) -> tensor<3x2xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [1], start_index_map = [1], index_vector_dim = 0>,
    indices_are_sorted = false
  } : (tensor<3x8xi32>, tensor<1x3x2xi64>, tensor<2xi64>) -> tensor<3x2xi32>
  return %r : tensor<3x2xi32>
}


// -----

// Signed narrow indices widen before clamping, so an i8 index can address a
// large operand dimension without narrowing the clamp bound.
// CHECK-LABEL: @dynamic_gather_narrow_indices
// CHECK: arith.index_cast %{{.+}} : i8 to index
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: return %{{.+}} : tensor<2x1xi32>
func.func @dynamic_gather_narrow_indices(%a: tensor<256x1xi32>, %i: tensor<2x1xi8>, %s: tensor<2xi64>) -> tensor<2x1xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false
  } : (tensor<256x1xi32>, tensor<2x1xi8>, tensor<2xi64>) -> tensor<2x1xi32>
  return %r : tensor<2x1xi32>
}


// -----

// Unsigned full-width indices clamp in their unsigned type before conversion
// to index, preserving ui64 values above the signed range.
// CHECK-LABEL: @dynamic_gather_unsigned_indices
// CHECK: arith.minui
// CHECK: arith.index_castui
// CHECK: return %{{.+}} : tensor<2x1xi32>
func.func @dynamic_gather_unsigned_indices(%a: tensor<5x1xi32>, %i: tensor<2x1xui64>, %s: tensor<2xi64>) -> tensor<2x1xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false
  } : (tensor<5x1xi32>, tensor<2x1xui64>, tensor<2xi64>) -> tensor<2x1xi32>
  return %r : tensor<2x1xi32>
}


// -----

// Clamp signed i64 gathers before backend index-width reduction.
// CHECK-LABEL: @gather_wide_signed_indices
// CHECK: arith.maxsi {{.*}} : i64
// CHECK: arith.minsi {{.*}} : i64
// CHECK: arith.index_cast {{.*}} : i64 to index
func.func @gather_wide_signed_indices(%a: tensor<3x4xi32>, %i: tensor<2x1xi64>) -> tensor<2x4xi32> {
  %r = "stablehlo.gather"(%a, %i) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<3x4xi32>, tensor<2x1xi64>) -> tensor<2x4xi32>
  return %r : tensor<2x4xi32>
}

// Unsigned gathers use the same source-width clamp, including values whose
// high bit would otherwise look negative after an index-width conversion.
// CHECK-LABEL: @gather_unsigned_indices
// CHECK: arith.minui {{.*}} : i64
// CHECK: arith.index_cast {{.*}} : i64 to index
func.func @gather_unsigned_indices(%a: tensor<3x4xi32>, %i: tensor<2x1xui64>) -> tensor<2x4xi32> {
  %r = "stablehlo.gather"(%a, %i) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<3x4xi32>, tensor<2x1xui64>) -> tensor<2x4xi32>
  return %r : tensor<2x4xi32>
}

// Signless i32 inputs widen before clamping, preserving their signed semantics.
// CHECK-LABEL: @gather_signless_i32_indices
// CHECK: arith.extsi {{.*}} : i32 to i64
// CHECK: arith.maxsi {{.*}} : i64
// CHECK: arith.minsi {{.*}} : i64
// CHECK: arith.index_cast {{.*}} : i64 to index
func.func @gather_signless_i32_indices(%a: tensor<3x4xi32>, %i: tensor<2x1xi32>) -> tensor<2x4xi32> {
  %r = "stablehlo.gather"(%a, %i) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<3x4xi32>, tensor<2x1xi32>) -> tensor<2x4xi32>
  return %r : tensor<2x4xi32>
}


// Dynamic operand shapes use the same typed clamp before index conversion.
// CHECK-LABEL: @gather_wide_dynamic_operand
// CHECK: tensor.dim %{{.*}}, %c0
// CHECK: arith.maxsi {{.*}} : i64
// CHECK: arith.minsi {{.*}} : i64
// CHECK: arith.index_cast {{.*}} : i64 to index
func.func @gather_wide_dynamic_operand(%a: tensor<?x4xi32>, %i: tensor<2x1xi64>) -> tensor<2x4xi32> {
  %r = "stablehlo.gather"(%a, %i) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<?x4xi32>, tensor<2x1xi64>) -> tensor<2x4xi32>
  return %r : tensor<2x4xi32>
}

// CHECK-LABEL: @gather_unsigned_dynamic_operand
// CHECK: tensor.dim %{{.*}}, %c0
// CHECK: arith.minui {{.*}} : i64
// CHECK: arith.index_castui {{.*}} : i64 to index
func.func @gather_unsigned_dynamic_operand(%a: tensor<?x4xi32>, %i: tensor<2x1xui64>) -> tensor<2x4xi32> {
  %r = "stablehlo.gather"(%a, %i) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<?x4xi32>, tensor<2x1xui64>) -> tensor<2x4xi32>
  return %r : tensor<2x4xi32>
}

// -----

// Dynamic output dimensions use the dynamic input's runtime extent.
// CHECK-LABEL: @dynamic_reduce_window_strided
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?xi32>
// CHECK: tensor.dim %[[ARG0]], %c0
// CHECK: linalg.generic
// CHECK: arith.addi
// CHECK: return %{{.+}} : tensor<?xi32>
func.func @dynamic_reduce_window_strided(%a: tensor<?xi32>, %init: tensor<i32>) -> tensor<?xi32> {
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<i32>
    "stablehlo.return"(%sum) : (tensor<i32>) -> ()
  }) {window_dimensions = array<i64: 2>, window_strides = array<i64: 2>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}


// -----

// A window larger than the runtime input has a zero-sized result.
// CHECK-LABEL: @dynamic_reduce_window_empty
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: linalg.generic
func.func @dynamic_reduce_window_empty(%a: tensor<?xi32>, %init: tensor<i32>) -> tensor<?xi32> {
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<i32>
    "stablehlo.return"(%sum) : (tensor<i32>) -> ()
  }) {window_dimensions = array<i64: 3>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}


// -----

// Padding and base dilation affect the runtime output extent and values.
// CHECK-LABEL: @dynamic_reduce_window_pad_base_dilation
// CHECK: tensor.insert_slice %[[ARG0:.+]] into %{{.+}}[1] [%{{.+}}] [2] : tensor<?xi32> into tensor<?xi32>
// CHECK: arith.cmpi sge
// CHECK: linalg.generic
func.func @dynamic_reduce_window_pad_base_dilation(%a: tensor<?xi32>, %init: tensor<i32>) -> tensor<?xi32> {
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<i32>
    "stablehlo.return"(%sum) : (tensor<i32>) -> ()
  }) {window_dimensions = array<i64: 3>, window_strides = array<i64: 2>, base_dilations = array<i64: 2>, padding = dense<[[1, 1]]> : tensor<1x2xi64>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}


// -----

// Window dilation changes the effective window size in the shape formula.
// CHECK-LABEL: @dynamic_reduce_window_window_dilation
// CHECK: linalg.generic
// CHECK: return %{{.+}} : tensor<?xi32>
func.func @dynamic_reduce_window_window_dilation(%a: tensor<?xi32>, %init: tensor<i32>) -> tensor<?xi32> {
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<i32>
    "stablehlo.return"(%sum) : (tensor<i32>) -> ()
  }) {window_dimensions = array<i64: 2>, window_strides = array<i64: 1>, window_dilations = array<i64: 2>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}


// -----

// Multi-result lowering shares one generic iteration space while preserving
// each result's reducer and initial value.
// CHECK-LABEL: @dynamic_reduce_window_multi_result
// CHECK: linalg.generic
// CHECK: arith.addi
// CHECK: arith.subi
func.func @dynamic_reduce_window_multi_result(%a: tensor<?xi32>, %b: tensor<?xi32>, %init0: tensor<i32>, %init1: tensor<i32>) -> (tensor<?xi32>, tensor<?xi32>) {
  %r:2 = "stablehlo.reduce_window"(%a, %b, %init0, %init1) ({
  ^bb0(%lhs0: tensor<i32>, %lhs1: tensor<i32>, %rhs0: tensor<i32>, %rhs1: tensor<i32>):
    %sum = stablehlo.add %lhs0, %rhs0 : tensor<i32>
    %difference = stablehlo.subtract %lhs1, %rhs1 : tensor<i32>
    "stablehlo.return"(%sum, %difference) : (tensor<i32>, tensor<i32>) -> ()
  }) {window_dimensions = array<i64: 1>} : (tensor<?xi32>, tensor<?xi32>, tensor<i32>, tensor<i32>) -> (tensor<?xi32>, tensor<?xi32>)
  return %r#0, %r#1 : tensor<?xi32>, tensor<?xi32>
}


// -----

// Per-dim window, stride and padding: only the dynamic dim takes the runtime
// output formula.
// CHECK-LABEL: @dynamic_reduce_window_rank2
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x7xi32>
// CHECK: %[[PADDED:.+]] = tensor.pad %[[ARG0]] low[1, 0] high[2, 3]
// CHECK: } : tensor<?x7xi32> to tensor<?x10xi32>
// CHECK: %[[DIM:.+]] = tensor.dim %[[PADDED]], %c0 : tensor<?x10xi32>
// CHECK: %[[SPAN:.+]] = arith.subi %[[DIM]], %c3 : index
// CHECK: %[[QUOTIENT:.+]] = arith.divsi %[[SPAN]], %c2 : index
// CHECK: tensor.empty(%{{.+}}) : tensor<?x9xi32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[PADDED]], %{{.+}} : tensor<?x10xi32>, tensor<3x2xi32>)
// CHECK: return %{{.+}} : tensor<?x9xi32>
func.func @dynamic_reduce_window_rank2(%a: tensor<?x7xi32>, %init: tensor<i32>) -> tensor<?x9xi32> {
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<i32>
    "stablehlo.return"(%sum) : (tensor<i32>) -> ()
  }) {window_dimensions = array<i64: 3, 2>, window_strides = array<i64: 2, 1>, padding = dense<[[1, 2], [0, 3]]> : tensor<2x2xi64>} : (tensor<?x7xi32>, tensor<i32>) -> tensor<?x9xi32>
  return %r : tensor<?x9xi32>
}


// -----

// A dynamic batch keeps the named pooling op; only spatial dims need be static.
// CHECK-LABEL: func.func @dynamic_batch_max_pool
// CHECK: linalg.pooling_nhwc_max
// CHECK-SAME: strides = dense<[2, 3]> : vector<2xi64>
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<?x13x11x5xf32>, tensor<3x2xf32>)
func.func @dynamic_batch_max_pool(%input: tensor<?x13x11x5xf32>) -> tensor<?x6x4x5xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%input, %init) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %max = stablehlo.maximum %a, %b : tensor<f32>
    stablehlo.return %max : tensor<f32>
  }) {window_dimensions = array<i64: 1, 3, 2, 1>,
      window_strides = array<i64: 1, 2, 3, 1>}
    : (tensor<?x13x11x5xf32>, tensor<f32>) -> tensor<?x6x4x5xf32>
  return %0 : tensor<?x6x4x5xf32>
}
