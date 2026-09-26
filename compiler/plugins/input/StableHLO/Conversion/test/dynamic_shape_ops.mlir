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

// CHECK-LABEL: @dynamic_gather
// CHECK: linalg.generic
// CHECK: tensor.extract %{{.+}} : tensor<3x4x2xi32>
// CHECK: return %{{.+}} : tensor<2x3x2x2xi32>
func.func @dynamic_gather(%a: tensor<3x4x2xi32>, %i: tensor<2x3x2xi64>, %s: tensor<3xi64>) -> tensor<2x3x2x2xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi32>
  return %r : tensor<2x3x2x2xi32>
}

// -----

// A dynamic result: the offset dims come from the slice_sizes operand.
// CHECK-LABEL: @dynamic_gather_dynamic_result
// CHECK: tensor.empty(%{{.+}}, %{{.+}}) : tensor<2x3x?x?xi32>
// CHECK: linalg.generic
// CHECK: return %{{.+}} : tensor<2x3x?x?xi32>
func.func @dynamic_gather_dynamic_result(%a: tensor<3x4x2xi32>, %i: tensor<2x3x2xi64>, %s: tensor<3xi64>) -> tensor<2x3x?x?xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x?x?xi32>
  return %r : tensor<2x3x?x?xi32>
}

// -----

// Every operand dim distinct. The clamp bound per start dim is
// operand_dim - slice_size: 5 - 1 for the collapsed dim, 8 - 3 for the
// offset dim of size 3. i32 indices.
// CHECK-LABEL: @dynamic_gather_distinct
// CHECK-DAG: arith.constant 4 : index
// CHECK-DAG: arith.constant 5 : index
// CHECK: linalg.generic
// CHECK: tensor.extract %{{.+}} : tensor<5x8x2xi32>
// CHECK: return %{{.+}} : tensor<2x7x3x2xi32>
func.func @dynamic_gather_distinct(%a: tensor<5x8x2xi32>, %i: tensor<2x7x2xi32>, %s: tensor<3xi64>) -> tensor<2x7x3x2xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<5x8x2xi32>, tensor<2x7x2xi32>, tensor<3xi64>) -> tensor<2x7x3x2xi32>
  return %r : tensor<2x7x3x2xi32>
}

// -----

// A dynamic operand: the clamp bound for the dynamic start dim is read at
// runtime. Dim 2 is not a start dim and gets no clamp.
// CHECK-LABEL: @dynamic_gather_dynamic_operand
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x8x?xi32>,
// CHECK: linalg.generic
// CHECK: %[[D0:.+]] = tensor.dim %[[ARG0]], %c0
// CHECK: arith.subi %[[D0]], %c1
// CHECK-NOT: tensor.dim
// CHECK: return %{{.+}} : tensor<2x7x3x2xi32>
func.func @dynamic_gather_dynamic_operand(%a: tensor<?x8x?xi32>, %i: tensor<2x7x2xi64>, %s: tensor<3xi64>) -> tensor<2x7x3x2xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<?x8x?xi32>, tensor<2x7x2xi64>, tensor<3xi64>) -> tensor<2x7x3x2xi32>
  return %r : tensor<2x7x3x2xi32>
}

// -----

// A dynamic batch dim in the indices flows into the result.
// CHECK-LABEL: @dynamic_gather_dynamic_indices
// CHECK-SAME: (%[[ARG0:.+]]: tensor<5x8x2xi32>, %[[IDX:.+]]: tensor<2x?x2xi64>,
// CHECK: %[[N:.+]] = tensor.dim %[[IDX]], %c1
// CHECK: tensor.empty(%[[N]]) : tensor<2x?x3x2xi32>
// CHECK: linalg.generic
// CHECK: return %{{.+}} : tensor<2x?x3x2xi32>
func.func @dynamic_gather_dynamic_indices(%a: tensor<5x8x2xi32>, %i: tensor<2x?x2xi64>, %s: tensor<3xi64>) -> tensor<2x?x3x2xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<5x8x2xi32>, tensor<2x?x2xi64>, tensor<3xi64>) -> tensor<2x?x3x2xi32>
  return %r : tensor<2x?x3x2xi32>
}

// -----

// index_vector_dim equal to the indices rank: one scalar index per element.
// The collapsed dim 1 has bound 8 - 1.
// CHECK-LABEL: @dynamic_gather_implicit_index_vector
// CHECK-DAG: arith.constant 7 : index
// CHECK: linalg.generic
// CHECK: return %{{.+}} : tensor<2x7x5x2xi32>
func.func @dynamic_gather_implicit_index_vector(%a: tensor<5x8x2xi32>, %i: tensor<2x7xi64>, %s: tensor<3xi64>) -> tensor<2x7x5x2xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<5x8x2xi32>, tensor<2x7xi64>, tensor<3xi64>) -> tensor<2x7x5x2xi32>
  return %r : tensor<2x7x5x2xi32>
}

// -----

// Batching dims pair operand dim 0 with indices dim 0.
// CHECK-LABEL: @dynamic_gather_batching
// CHECK: linalg.generic
// CHECK: return %{{.+}} : tensor<3x7x5xi32>
func.func @dynamic_gather_batching(%a: tensor<3x8x5xi32>, %i: tensor<3x7x1xi64>, %s: tensor<3xi64>) -> tensor<3x7x5xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<3x8x5xi32>, tensor<3x7x1xi64>, tensor<3xi64>) -> tensor<3x7x5xi32>
  return %r : tensor<3x7x5xi32>
}

// -----

// Runtime expansion depends on the operand shape, not the output index.
// CHECK-LABEL: @dynamic_broadcast_undecidable
// CHECK: %[[EXPANDS:.+]] = arith.cmpi eq
// CHECK: linalg.generic
// CHECK-NOT: arith.cmpi
// CHECK: arith.select %[[EXPANDS]]
// CHECK: tensor.extract %{{.+}} : tensor<?xf32>
// CHECK: return %{{.+}} : tensor<?x?xf32>
func.func @dynamic_broadcast_undecidable(%a: tensor<?xf32>, %s: tensor<2xi32>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [1] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// The annotated form decides expansion statically, so upstream's own lowering applies.
// CHECK-LABEL: @dynamic_broadcast_annotated
// CHECK: linalg.generic
// CHECK-NOT: arith.select
// CHECK-NOT: tensor.extract
// CHECK: return %{{.+}} : tensor<?x?xf32>
func.func @dynamic_broadcast_annotated(%a: tensor<?xf32>, %s: tensor<2xi32>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [1] {known_nonexpanding_dimensions = array<i64: 0>} : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// Rank 3 into rank 4: the two dynamic operand dims each get a select, the
// static 1 indexes zero without one.
// CHECK-LABEL: @dynamic_broadcast_mixed
// CHECK-COUNT-2: arith.cmpi eq
// CHECK: linalg.generic
// CHECK-NOT: arith.cmpi
// CHECK-COUNT-2: arith.select
// CHECK-NOT: arith.select
// CHECK: tensor.extract %{{.+}} : tensor<?x1x?xf32>
// CHECK: return %{{.+}} : tensor<?x?x?x?xf32>
func.func @dynamic_broadcast_mixed(%a: tensor<?x1x?xf32>, %s: tensor<4xi32>) -> tensor<?x?x?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [1, 2, 3] : (tensor<?x1x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  return %r : tensor<?x?x?x?xf32>
}

// -----

// A static non-1 dim indexes directly; only the dynamic dim gets a select.
// CHECK-LABEL: @dynamic_broadcast_static_non1
// CHECK: linalg.generic
// CHECK: arith.select
// CHECK-NOT: arith.select
// CHECK: tensor.extract %{{.+}} : tensor<3x?xf32>
func.func @dynamic_broadcast_static_non1(%a: tensor<3x?xf32>, %s: tensor<2xi32>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [0, 1] : (tensor<3x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// Permuted broadcast_dimensions with an i64 shape: operand dim 0 reads result
// index 2 and operand dim 1 reads result index 0.
// CHECK-LABEL: @dynamic_broadcast_permuted
// CHECK: linalg.generic
// CHECK-DAG: linalg.index 2
// CHECK-DAG: linalg.index 0
// CHECK-NOT: linalg.index 1
// CHECK: tensor.extract %{{.+}} : tensor<?x?xf32>
// CHECK: return %{{.+}} : tensor<?x?x?xf32>
func.func @dynamic_broadcast_permuted(%a: tensor<?x?xf32>, %s: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [2, 0] : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// One dynamic dim annotated, one not: the unannotated dim keeps this path.
// CHECK-LABEL: @dynamic_broadcast_partial_annotation
// CHECK: %[[EXPANDS:.+]] = arith.cmpi eq
// CHECK-NOT: arith.cmpi
// CHECK: linalg.generic
// CHECK-NOT: arith.cmpi
// CHECK: arith.select %[[EXPANDS]]
// CHECK-NOT: arith.select
// CHECK: tensor.extract %{{.+}} : tensor<?x?xf32>
func.func @dynamic_broadcast_partial_annotation(%a: tensor<?x?xf32>, %s: tensor<2xi32>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [0, 1] {known_nonexpanding_dimensions = array<i64: 0>} : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// Both dynamic dims annotated, one each way: decidable, upstream path.
// CHECK-LABEL: @dynamic_broadcast_annotated_both
// CHECK: linalg.generic
// CHECK-NOT: arith.select
// CHECK-NOT: tensor.extract
// CHECK: return %{{.+}} : tensor<?x?x?xf32>
func.func @dynamic_broadcast_annotated_both(%a: tensor<?x?xf32>, %s: tensor<3xi32>) -> tensor<?x?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [1, 2] {known_expanding_dimensions = array<i64: 0>, known_nonexpanding_dimensions = array<i64: 1>} : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// Removing a leading index-vector dimension shifts the batching dimension down.
// CHECK-LABEL: @dynamic_gather_leading_index_vector
// CHECK: linalg.generic
// CHECK: return %{{.+}} : tensor<3x2xf32>
func.func @dynamic_gather_leading_index_vector(%a: tensor<3x8xf32>, %i: tensor<1x3x2xi64>, %s: tensor<2xi64>) -> tensor<3x2xf32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [1], start_index_map = [1], index_vector_dim = 0>,
    indices_are_sorted = false
  } : (tensor<3x8xf32>, tensor<1x3x2xi64>, tensor<2xi64>) -> tensor<3x2xf32>
  return %r : tensor<3x2xf32>
}
