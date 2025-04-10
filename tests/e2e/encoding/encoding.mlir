//===----------------------------------------------------------------------===//
// Utility Methods
//===----------------------------------------------------------------------===//

func.func private @generate_1D_source_f32(%height : index) -> tensor<?xf32> {
  %init_source = tensor.empty(%height) : tensor<?xf32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : tensor<?xf32>) {
    ^bb0(%b0 : f32):
      %index = linalg.index 0 : index
      %index_i32 = arith.index_cast %index : index to i32
      %index_f32 = arith.sitofp %index_i32 : i32 to f32
      linalg.yield %index_f32 : f32
  } -> tensor<?xf32>
  // This blocks the fusion for inputs and testing ops.
  %0 = util.optimization_barrier %source : tensor<?xf32>
  %1 = flow.tensor.tie_shape %0 : tensor<?xf32>{%height}
  return %1 : tensor<?xf32>
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

func.func private @generate_3D_source_f32(%height : index, %width : index, %depth : index) -> tensor<?x?x?xf32> {
  %init_source = tensor.empty(%height, %width, %depth) : tensor<?x?x?xf32>
  %wd = arith.muli %width, %depth : index
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      outs(%init_source : tensor<?x?x?xf32>) {
    ^bb0(%b0 : f32):
      %h = linalg.index 0 : index
      %w = linalg.index 1 : index
      %d = linalg.index 2 : index
      %strided_h = arith.muli %h, %wd : index
      %strided_w = arith.muli %w, %depth : index
      %strided_wd = arith.addi %strided_w, %d : index
      %linearized = arith.addi %strided_h, %strided_wd : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      %linearized_f32 = arith.sitofp %linearized_i32 : i32 to f32
      linalg.yield %linearized_f32 : f32
  } -> tensor<?x?x?xf32>
  // This blocks the fusion for inputs and testing ops.
  %0 = util.optimization_barrier %source : tensor<?x?x?xf32>
  %1 = flow.tensor.tie_shape %0 : tensor<?x?x?xf32>{%height, %width, %depth}
  return %1 : tensor<?x?x?xf32>
}

//===----------------------------------------------------------------------===//
// Elementwise materialization tests
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#bcast_map_1d = affine_map<(d0, d1) -> (d1)>
#encoding_f32f32f32_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_f32f32f32_lhs_bcast = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[#map, #bcast_map_1d], #map1, #map2]>

func.func @elementwise() {
  %height = arith.constant 129 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source_f32(%height, %width) : (index, index) -> tensor<?x?xf32>
  %source = tensor.cast %0 : tensor<?x?xf32> to tensor<129x255xf32>

  %1 = iree_encoding.set_encoding %source : tensor<129x255xf32> -> tensor<129x255xf32, #encoding_f32f32f32_lhs>
  %2 = tensor.empty() : tensor<129x255xf32, #encoding_f32f32f32_lhs>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<129x255xf32, #encoding_f32f32f32_lhs>, tensor<129x255xf32, #encoding_f32f32f32_lhs>) outs(%2 : tensor<129x255xf32, #encoding_f32f32f32_lhs>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<129x255xf32, #encoding_f32f32f32_lhs>
  %barrier = util.optimization_barrier %3 : tensor<129x255xf32, #encoding_f32f32f32_lhs>
  %5 = iree_encoding.unset_encoding %3 : tensor<129x255xf32, #encoding_f32f32f32_lhs> -> tensor<129x255xf32>

  %expected = arith.addf %source, %source : tensor<129x255xf32>
  check.expect_almost_eq(%5, %expected) : tensor<129x255xf32>
  return
}

func.func @elementwise_with_broadcast_2d() {
  %height = arith.constant 128 : index
  %width = arith.constant 64 : index
  %0 = call @generate_2D_source_f32(%height, %width) : (index, index) -> tensor<?x?xf32>
  %source = tensor.cast %0 : tensor<?x?xf32> to tensor<128x64xf32>
  %1 = call @generate_1D_source_f32(%width) : (index) -> tensor<?xf32>
  %bcast = tensor.cast %1 : tensor<?xf32> to tensor<64xf32>

  %2 = iree_encoding.set_encoding %source : tensor<128x64xf32> -> tensor<128x64xf32, #encoding_f32f32f32_lhs>
  %3 = iree_encoding.set_encoding %bcast : tensor<64xf32> -> tensor<64xf32, #encoding_f32f32f32_lhs_bcast>
  %4 = tensor.empty() : tensor<128x64xf32, #encoding_f32f32f32_lhs>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<128x64xf32, #encoding_f32f32f32_lhs>,tensor<64xf32, #encoding_f32f32f32_lhs_bcast>) outs(%4 : tensor<128x64xf32, #encoding_f32f32f32_lhs>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %6 = arith.addf %in, %b : f32
    linalg.yield %6 : f32
  } -> tensor<128x64xf32, #encoding_f32f32f32_lhs>
  %barrier = util.optimization_barrier %5 : tensor<128x64xf32, #encoding_f32f32f32_lhs>
  %6 = iree_encoding.unset_encoding %5 : tensor<128x64xf32, #encoding_f32f32f32_lhs> -> tensor<128x64xf32>

  %init_expected = tensor.empty() : tensor<128x64xf32>
  %expected =  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%source, %bcast : tensor<128x64xf32>, tensor<64xf32>) outs(%init_expected : tensor<128x64xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %7 = arith.addf %in, %b : f32
    linalg.yield %7 : f32
  } -> tensor<128x64xf32>
  check.expect_almost_eq(%6, %expected) : tensor<128x64xf32>
  return
}

#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#bcast_map_2d = affine_map<(d0, d1, d2) -> (d0, d2)>
#encoding_3d_f32f32f32_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map3, #map4, #map5]>
#encoding_3d_f32f32f32_lhs_bcast = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[#map3, #bcast_map_2d], #map4, #map5]>
func.func @elementwise_with_broadcast_3d() {
  %height = arith.constant 2 : index
  %width = arith.constant 128 : index
  %depth = arith.constant 64 : index
  %0 = call @generate_3D_source_f32(%height, %width, %depth) : (index, index, index) -> tensor<?x?x?xf32>
  %source = tensor.cast %0 : tensor<?x?x?xf32> to tensor<2x128x64xf32>
  %1 = call @generate_2D_source_f32(%height, %depth) : (index, index) -> tensor<?x?xf32>
  %bcast = tensor.cast %1 : tensor<?x?xf32> to tensor<2x64xf32>

  %2 = iree_encoding.set_encoding %source : tensor<2x128x64xf32> -> tensor<2x128x64xf32, #encoding_3d_f32f32f32_lhs>
  %3 = iree_encoding.set_encoding %bcast : tensor<2x64xf32> -> tensor<2x64xf32, #encoding_3d_f32f32f32_lhs_bcast>
  %4 = tensor.empty() : tensor<2x128x64xf32, #encoding_3d_f32f32f32_lhs>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<2x128x64xf32, #encoding_3d_f32f32f32_lhs>, tensor<2x64xf32, #encoding_3d_f32f32f32_lhs_bcast>) outs(%4 : tensor<2x128x64xf32, #encoding_3d_f32f32f32_lhs>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %24 = arith.addf %in, %b : f32
    linalg.yield %24 : f32
  } -> tensor<2x128x64xf32, #encoding_3d_f32f32f32_lhs>
  %barrier = util.optimization_barrier %5 : tensor<2x128x64xf32, #encoding_3d_f32f32f32_lhs>
  %6 = iree_encoding.unset_encoding %5 : tensor<2x128x64xf32, #encoding_3d_f32f32f32_lhs> -> tensor<2x128x64xf32>

  %init_expected = tensor.empty() : tensor<2x128x64xf32>
  %expected =  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%source, %bcast : tensor<2x128x64xf32>, tensor<2x64xf32>) outs(%init_expected : tensor<2x128x64xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %7 = arith.addf %in, %b : f32
    linalg.yield %7 : f32
  } -> tensor<2x128x64xf32>
  check.expect_almost_eq(%6, %expected) : tensor<2x128x64xf32>
  return
}
