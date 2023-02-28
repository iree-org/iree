func.func private @generate_2D_source(%height : index, %width : index) -> tensor<?x?xi32> {
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
  return %source : tensor<?x?xi32>
}

func.func @fully_dynamic_pack_simple() {
  %iree_input = flow.tensor.constant dense<[
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15]]> : tensor<4x4xi32> -> tensor<?x?xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = util.unfoldable_constant 2 : index
  %in_d0 = tensor.dim %iree_input, %c0 : tensor<?x?xi32>
  %in_d1 = tensor.dim %iree_input, %c1 : tensor<?x?xi32>
  %out_d0 = arith.ceildivui %in_d0, %c2 : index
  %out_d1 = arith.ceildivui %in_d1, %c2 : index
  %init = tensor.empty(%out_d0, %out_d1, %c2, %c2) : tensor<?x?x?x?xi32>
  %pack = tensor.pack %iree_input inner_dims_pos = [0, 1] inner_tiles = [%c2, %c2] into %init
      : tensor<?x?xi32> -> tensor<?x?x?x?xi32>
  %cast = tensor.cast %pack : tensor<?x?x?x?xi32> to tensor<2x2x2x2xi32>
  check.expect_eq_const(%cast, dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32>) : tensor<2x2x2x2xi32>
  return
}

func.func @fully_dynamic_pack_pad_transpose_inner_and_outer_dims_large() {
  %d0 = util.unfoldable_constant 100 : index
  %d1 = util.unfoldable_constant 250 : index
  %source = call @generate_2D_source(%d0, %d1) : (index, index) -> tensor<?x?xi32>
  %padding_value = arith.constant 42 : i32

  %c16 = util.unfoldable_constant 16 : index
  %c32 = util.unfoldable_constant 32 : index
  %tiled_d0 = arith.ceildivui %d0, %c32 : index
  %tiled_d1 = arith.ceildivui %d1, %c16 : index
  %init_pack = tensor.empty(%tiled_d1, %tiled_d0, %c16, %c32) : tensor<?x?x?x?xi32>
  %pack = tensor.pack %source padding_value(%padding_value : i32)
      outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [%c16, %c32] into %init_pack
      : tensor<?x?xi32> -> tensor<?x?x?x?xi32>
  %cast_pack = tensor.cast %pack : tensor<?x?x?x?xi32> to tensor<16x4x16x32xi32>

  %c100 = arith.constant 100 : index
  %c250 = arith.constant 250 : index
  %source2 = call @generate_2D_source(%c100, %c250) : (index, index) -> tensor<?x?xi32>
  %static_source = tensor.cast %source2 : tensor<?x?xi32> to tensor<100x250xi32>

  %pad = tensor.pad %static_source low[0, 0] high[28, 6] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %padding_value : i32
  } : tensor<100x250xi32> to tensor<128x256xi32>
  %reshape = tensor.expand_shape %pad [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = tensor.empty() : tensor<16x4x16x32xi32>
  %transpose = linalg.transpose
    ins(%reshape : tensor<4x32x16x16xi32>)
    outs(%init_transpose : tensor<16x4x16x32xi32>)
    permutation = [2, 0, 3, 1]

  check.expect_eq(%cast_pack, %transpose) : tensor<16x4x16x32xi32>
  return
}
