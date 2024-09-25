//===----------------------------------------------------------------------===//
// Utility Methods
//===----------------------------------------------------------------------===//

func.func private @generate_2D_source_f16(%height : index, %width : index) -> tensor<?x?xf16> {
  %init_source = tensor.empty(%height, %width) : tensor<?x?xf16>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init_source : tensor<?x?xf16>) {
    ^bb0(%b0 : f16):
      %outer = linalg.index 0 : index
      %inner = linalg.index 1 : index
      %strided = arith.muli %outer, %width : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i16 = arith.index_cast %linearized : index to i16
      %linearized_f16 = arith.sitofp %linearized_i16 : i16 to f16
      linalg.yield %linearized_f16 : f16
  } -> tensor<?x?xf16>
  // This blocks the fusion for inputs and testing ops.
  %0 = util.optimization_barrier %source : tensor<?x?xf16>
  %1 = flow.tensor.tie_shape %0 : tensor<?x?xf16>{%height, %width}
  return %1 : tensor<?x?xf16>
}

func.func private @generate_2D_source_f32(%height : index, %width : index) -> tensor<?x?xf32> {
  %init_source = tensor.empty(%height, %width) : tensor<?x?xf32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init_source : tensor<?x?xf32>) {
    ^bb0(%b0 : f32):
      %outer = linalg.index 0 : index
      %inner = linalg.index 1 : index
      %strided = arith.muli %outer, %width : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      %linearized_f32 = arith.sitofp %linearized_i32 : i32 to f32
      linalg.yield %linearized_f32 : f32
  } -> tensor<?x?xf32>
  // This blocks the fusion for inputs and testing ops.
  %0 = util.optimization_barrier %source : tensor<?x?xf32>
  %1 = flow.tensor.tie_shape %0 : tensor<?x?xf32>{%height, %width}
  return %1 : tensor<?x?xf32>
}

func.func private @generate_2D_source_i8(%height : index, %width : index) -> tensor<?x?xi8> {
  %c255 = arith.constant 255 : index
  %init_source = tensor.empty(%height, %width) : tensor<?x?xi8>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init_source : tensor<?x?xi8>) {
    ^bb0(%b0 : i8):
      %outer = linalg.index 0 : index
      %inner = linalg.index 1 : index
      %strided = arith.muli %outer, %width : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_rem = arith.remsi %linearized, %c255 : index
      %linearized_i8 = arith.index_cast %linearized_rem : index to i8
      linalg.yield %linearized_i8 : i8
  } -> tensor<?x?xi8>
  // This blocks the fusion for inputs and testing ops.
  %0 = util.optimization_barrier %source : tensor<?x?xi8>
  %1 = flow.tensor.tie_shape %0 : tensor<?x?xi8>{%height, %width}
  return %1 : tensor<?x?xi8>
}

func.func private @generate_2D_source_i32(%height : index, %width : index) -> tensor<?x?xi32> {
  %init_source = tensor.empty(%height, %width) : tensor<?x?xi32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init_source : tensor<?x?xi32>) {
    ^bb0(%b0 : i32):
      %outer = linalg.index 0 : index
      %inner = linalg.index 1 : index
      %strided = arith.muli %outer, %width : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      linalg.yield %linearized_i32 : i32
  } -> tensor<?x?xi32>
  // This blocks the fusion for inputs and testing ops.
  %0 = util.optimization_barrier %source : tensor<?x?xi32>
  %1 = flow.tensor.tie_shape %0 : tensor<?x?xi32>{%height, %width}
  return %1 : tensor<?x?xi32>
}

//===----------------------------------------------------------------------===//
// f32.f32.f32 variants
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_f32f32f32_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_f32f32f32_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_f32f32f32_acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>

func.func @set_encoding_f32f32f32_lhs() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_f32(%height, %width) : (index, index) -> tensor<?x?xf32>
  %source = tensor.cast %0 : tensor<?x?xf32> to tensor<129x255xf32>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xf32> -> tensor<129x255xf32, #encoding_f32f32f32_lhs>
  %barrire = util.optimization_barrier %1 : tensor<129x255xf32, #encoding_f32f32f32_lhs>
  %2 = iree_encoding.unset_encoding %1 : tensor<129x255xf32, #encoding_f32f32f32_lhs> -> tensor<129x255xf32>
  check.expect_almost_eq(%2, %source) : tensor<129x255xf32>
  return
}

func.func @set_encoding_f32f32f32_rhs() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_f32(%height, %width) : (index, index) -> tensor<?x?xf32>
  %source = tensor.cast %0 : tensor<?x?xf32> to tensor<129x255xf32>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xf32> -> tensor<129x255xf32, #encoding_f32f32f32_rhs>
  %barrire = util.optimization_barrier %1 : tensor<129x255xf32, #encoding_f32f32f32_rhs>
  %2 = iree_encoding.unset_encoding %1 : tensor<129x255xf32, #encoding_f32f32f32_rhs> -> tensor<129x255xf32>
  check.expect_almost_eq(%2, %source) : tensor<129x255xf32>
  return
}

func.func @set_encoding_f32f32f32_acc() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_f32(%height, %width) : (index, index) -> tensor<?x?xf32>
  %source = tensor.cast %0 : tensor<?x?xf32> to tensor<129x255xf32>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xf32> -> tensor<129x255xf32, #encoding_f32f32f32_acc>
  %barrire = util.optimization_barrier %1 : tensor<129x255xf32, #encoding_f32f32f32_acc>
  %2 = iree_encoding.unset_encoding %1 : tensor<129x255xf32, #encoding_f32f32f32_acc> -> tensor<129x255xf32>
  check.expect_almost_eq(%2, %source) : tensor<129x255xf32>
  return
}

//===----------------------------------------------------------------------===//
// i8.i8.i32 variants
//===----------------------------------------------------------------------===//

#encoding_i8i8i32_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_i8i8i32_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_i8i8i32_acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>

func.func @set_encoding_i8i8i32_lhs() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_i8(%height, %width) : (index, index) -> tensor<?x?xi8>
  %source = tensor.cast %0 : tensor<?x?xi8> to tensor<129x255xi8>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xi8> -> tensor<129x255xi8, #encoding_i8i8i32_lhs>
  %barrire = util.optimization_barrier %1 : tensor<129x255xi8, #encoding_i8i8i32_lhs>
  %2 = iree_encoding.unset_encoding %1 : tensor<129x255xi8, #encoding_i8i8i32_lhs> -> tensor<129x255xi8>
  check.expect_eq(%2, %source) : tensor<129x255xi8>
  return
}

func.func @set_encoding_i8i8i32_rhs() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_i8(%height, %width) : (index, index) -> tensor<?x?xi8>
  %source = tensor.cast %0 : tensor<?x?xi8> to tensor<129x255xi8>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xi8> -> tensor<129x255xi8, #encoding_i8i8i32_rhs>
  %barrire = util.optimization_barrier %1 : tensor<129x255xi8, #encoding_i8i8i32_rhs>
  %2 = iree_encoding.unset_encoding %1 : tensor<129x255xi8, #encoding_i8i8i32_rhs> -> tensor<129x255xi8>
  check.expect_eq(%2, %source) : tensor<129x255xi8>
  return
}

func.func @set_encoding_i8i8i32_acc() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_i32(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<129x255xi32>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xi32> -> tensor<129x255xi32, #encoding_i8i8i32_acc>
  %barrire = util.optimization_barrier %1 : tensor<129x255xi32, #encoding_i8i8i32_acc>
  %2 = iree_encoding.unset_encoding %1 : tensor<129x255xi32, #encoding_i8i8i32_acc> -> tensor<129x255xi32>
  check.expect_eq(%2, %source) : tensor<129x255xi32>
  return
}


//===----------------------------------------------------------------------===//
// f16.f16.f32 variants
//===----------------------------------------------------------------------===//

#encoding_f16f16f32_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_f16f16f32_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_f16f16f32_acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>

func.func @set_encoding_f16f16f32_lhs() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_f16(%height, %width) : (index, index) -> tensor<?x?xf16>
  %source = tensor.cast %0 : tensor<?x?xf16> to tensor<129x255xf16>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xf16> -> tensor<129x255xf16, #encoding_f16f16f32_lhs>
  %barrire = util.optimization_barrier %1 : tensor<129x255xf16, #encoding_f16f16f32_lhs>
  %2 = iree_encoding.unset_encoding %1 : tensor<129x255xf16, #encoding_f16f16f32_lhs> -> tensor<129x255xf16>
  check.expect_eq(%2, %source) : tensor<129x255xf16>
  return
}

func.func @set_encoding_f16f16f32_rhs() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_f16(%height, %width) : (index, index) -> tensor<?x?xf16>
  %source = tensor.cast %0 : tensor<?x?xf16> to tensor<129x255xf16>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xf16> -> tensor<129x255xf16, #encoding_f16f16f32_rhs>
  %barrire = util.optimization_barrier %1 : tensor<129x255xf16, #encoding_f16f16f32_rhs>
  %2 = iree_encoding.unset_encoding %1 : tensor<129x255xf16, #encoding_f16f16f32_rhs> -> tensor<129x255xf16>
  check.expect_eq(%2, %source) : tensor<129x255xf16>
  return
}

func.func @set_encoding_f16f16f32_acc() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_f32(%height, %width) : (index, index) -> tensor<?x?xf32>
  %source = tensor.cast %0 : tensor<?x?xf32> to tensor<129x255xf32>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xf32> -> tensor<129x255xf32, #encoding_f16f16f32_acc>
  %barrire = util.optimization_barrier %1 : tensor<129x255xf32, #encoding_f16f16f32_acc>
  %2 = iree_encoding.unset_encoding %1 : tensor<129x255xf32, #encoding_f16f16f32_acc> -> tensor<129x255xf32>
  check.expect_eq(%2, %source) : tensor<129x255xf32>
  return
}
