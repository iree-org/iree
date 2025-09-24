func.func private @generate_4D_source(%d0: index, %d1: index, %d2: index, %d3: index) -> tensor<?x?x?x?xi32> {
  %init_source = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?xi32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      outs(%init_source : tensor<?x?x?x?xi32>) {
    ^bb0(%b0 : i32):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %k = linalg.index 2 : index
      %l = linalg.index 3 : index
      %t0 = arith.muli %i, %d1 : index
      %t1 = arith.addi %t0, %j : index
      %t2 = arith.muli %t1, %d2 : index
      %t3 = arith.addi %t2, %k : index
      %t4 = arith.muli %t3, %d3 : index
      %t5 = arith.addi %t4, %l : index
      %res = arith.index_cast %t5 : index to i32
      linalg.yield %res : i32
  } -> tensor<?x?x?x?xi32>
  return %source : tensor<?x?x?x?xi32>
}

func.func @static_unpack_simple() {
  %iree_input = util.unfoldable_constant dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32>
  %init = tensor.empty() : tensor<4x4xi32>
  %unpack = linalg.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : tensor<2x2x2x2xi32> -> tensor<4x4xi32>
  check.expect_eq_const(%unpack, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}

func.func @dynamic_unpack_simple() {
  %iree_input = flow.tensor.dynamic_constant dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32> -> tensor<?x?x2x2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %in_d0 = tensor.dim %iree_input, %c0 : tensor<?x?x2x2xi32>
  %in_d1 = tensor.dim %iree_input, %c1 : tensor<?x?x2x2xi32>
  %out_d0 = arith.muli %in_d0, %c2 : index
  %out_d1 = arith.muli %in_d1, %c2 : index
  %init = tensor.empty(%out_d0, %out_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : tensor<?x?x2x2xi32> -> tensor<?x?xi32>
  %cast = tensor.cast %unpack : tensor<?x?xi32> to tensor<4x4xi32>
  check.expect_eq_const(%cast, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}

func.func @static_unpack_simple_extract_slice() {
  %iree_input = util.unfoldable_constant dense<[[[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                                       [[3, 0, 0], [7, 0, 0], [11, 0, 0]]],
                                      [[[12, 13, 14], [0, 0, 0], [0, 0, 0]],
                                       [[15, 0, 0], [0, 0, 0], [0, 0, 0]]]]> : tensor<2x2x3x3xi32>
  %init = tensor.empty() : tensor<4x4xi32>
  %unpack = linalg.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %init
      : tensor<2x2x3x3xi32> -> tensor<4x4xi32>
  check.expect_eq_const(%unpack, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}

func.func @dynamic_unpack_simple_extract_slice() {
  %iree_input = flow.tensor.dynamic_constant dense<[[[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                                       [[3, 0, 0], [7, 0, 0], [11, 0, 0]]],
                                      [[[12, 13, 14], [0, 0, 0], [0, 0, 0]],
                                       [[15, 0, 0], [0, 0, 0], [0, 0, 0]]]]> : tensor<2x2x3x3xi32> -> tensor<?x?x3x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %in_d0 = tensor.dim %iree_input, %c0 : tensor<?x?x3x3xi32>
  %in_d1 = tensor.dim %iree_input, %c1 : tensor<?x?x3x3xi32>
  %full_out_d0 = arith.muli %in_d0, %c3 : index
  %full_out_d1 = arith.muli %in_d1, %c3 : index
  %out_d0 = arith.subi %full_out_d0, %c2 : index
  %out_d1 = arith.subi %full_out_d1, %c2 : index
  %init = tensor.empty(%out_d0, %out_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %init
      : tensor<?x?x3x3xi32> -> tensor<?x?xi32>
  %cast = tensor.cast %unpack : tensor<?x?xi32> to tensor<4x4xi32>
  check.expect_eq_const(%cast, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}

func.func @static_unpack_large() {
  %d0 = arith.constant 4 : index
  %d1 = arith.constant 16 : index
  %d2 = arith.constant 32 : index
  %d3 = arith.constant 16 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<4x16x32x16xi32>

  %init_unpack = tensor.empty() : tensor<128x256xi32>
  %unpack = linalg.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : tensor<4x16x32x16xi32> -> tensor<128x256xi32>

  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  %transpose = linalg.transpose
    ins(%source: tensor<4x16x32x16xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [0, 2, 1, 3]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>

  check.expect_eq(%unpack, %collapse) : tensor<128x256xi32>
  return
}

func.func @dynamic_unpack_large() {
  %d0 = util.unfoldable_constant 4 : index
  %d1 = util.unfoldable_constant 16 : index
  %d2 = util.unfoldable_constant 32 : index
  %d3 = util.unfoldable_constant 16 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<?x?x32x16xi32>

  %packed_d0 = util.unfoldable_constant 128 : index
  %packed_d1 = util.unfoldable_constant 256 : index
  %init_unpack = tensor.empty(%packed_d0, %packed_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
  %cast_unpack = tensor.cast %unpack : tensor<?x?xi32> to tensor<128x256xi32>

  %source2 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %static_source = tensor.cast %source2 : tensor<?x?x?x?xi32> to tensor<4x16x32x16xi32>
  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  %transpose = linalg.transpose
    ins(%static_source: tensor<4x16x32x16xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [0, 2, 1, 3]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>

  check.expect_eq(%cast_unpack, %collapse) : tensor<128x256xi32>
  return
}

func.func @dynamic_unpack_transpose_inner_dims_large() {
  %d0 = util.unfoldable_constant 4 : index
  %d1 = util.unfoldable_constant 16 : index
  %d2 = util.unfoldable_constant 16 : index
  %d3 = util.unfoldable_constant 32 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<?x?x16x32xi32>

  %packed_d0 = util.unfoldable_constant 128 : index
  %packed_d1 = util.unfoldable_constant 256 : index
  %init_unpack = tensor.empty(%packed_d0, %packed_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %source inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_unpack
      : tensor<?x?x16x32xi32> -> tensor<?x?xi32>
  %cast_unpack = tensor.cast %unpack : tensor<?x?xi32> to tensor<128x256xi32>

  %source2 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %static_source = tensor.cast %source2 : tensor<?x?x?x?xi32> to tensor<4x16x16x32xi32>
  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [0, 2, 3, 1] -> [0, 1, 2, 3] is [0, 3, 1, 2].
  %transpose = linalg.transpose
    ins(%static_source: tensor<4x16x16x32xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [0, 3, 1, 2]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>

  check.expect_eq(%cast_unpack, %collapse) : tensor<128x256xi32>
  return
}

func.func @dynamic_unpack_transpose_outer_dims_large() {
  %d0 = util.unfoldable_constant 16 : index
  %d1 = util.unfoldable_constant 4 : index
  %d2 = util.unfoldable_constant 32 : index
  %d3 = util.unfoldable_constant 16 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<?x?x32x16xi32>

  %packed_d0 = util.unfoldable_constant 128 : index
  %packed_d1 = util.unfoldable_constant 256 : index
  %init_unpack = tensor.empty(%packed_d0, %packed_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
  %cast_unpack = tensor.cast %unpack : tensor<?x?xi32> to tensor<128x256xi32>

  %source2 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %static_source = tensor.cast %source2 : tensor<?x?x?x?xi32> to tensor<16x4x32x16xi32>
  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [2, 0, 1, 3] -> [0, 1, 2, 3] is [1, 2, 0, 3].
  %transpose = linalg.transpose
    ins(%static_source: tensor<16x4x32x16xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [1, 2, 0, 3]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>

  check.expect_eq(%cast_unpack, %collapse) : tensor<128x256xi32>
  return
}

func.func @dynamic_unpack_transpose_inner_and_outer_dims_large() {
  %d0 = util.unfoldable_constant 16 : index
  %d1 = util.unfoldable_constant 4 : index
  %d2 = util.unfoldable_constant 16 : index
  %d3 = util.unfoldable_constant 32 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<?x?x16x32xi32>

  %packed_d0 = util.unfoldable_constant 128 : index
  %packed_d1 = util.unfoldable_constant 256 : index
  %init_unpack = tensor.empty(%packed_d0, %packed_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %source outer_dims_perm = [1, 0]  inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_unpack
      : tensor<?x?x16x32xi32> -> tensor<?x?xi32>
  %cast_unpack = tensor.cast %unpack : tensor<?x?xi32> to tensor<128x256xi32>

  %source2 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %static_source = tensor.cast %source2 : tensor<?x?x?x?xi32> to tensor<16x4x16x32xi32>
  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [2, 0, 3, 1] -> [0, 1, 2, 3] is [1, 3, 0, 2].
  %transpose = linalg.transpose
    ins(%static_source: tensor<16x4x16x32xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [1, 3, 0, 2]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>

  check.expect_eq(%cast_unpack, %collapse) : tensor<128x256xi32>
  return
}

func.func @static_unpack_extract_slice_large() {
  %d0 = arith.constant 4 : index
  %d1 = arith.constant 16 : index
  %d2 = arith.constant 32 : index
  %d3 = arith.constant 16 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<4x16x32x16xi32>

  %init_unpack = tensor.empty() : tensor<100x250xi32>
  %unpack = linalg.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : tensor<4x16x32x16xi32> -> tensor<100x250xi32>

  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [0, 2, 1, 3] -> [0, 1, 2, 3] is [0, 2, 1, 3].
  %transpose = linalg.transpose
    ins(%source: tensor<4x16x32x16xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [0, 2, 1, 3]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>
  %slice = tensor.extract_slice %collapse [0, 0] [100, 250] [1, 1] : tensor<128x256xi32> to tensor<100x250xi32>

  check.expect_eq(%unpack, %slice) : tensor<100x250xi32>
  return
}

func.func @dynamic_unpack_extract_slice_large() {
  %d0 = util.unfoldable_constant 4 : index
  %d1 = util.unfoldable_constant 16 : index
  %d2 = util.unfoldable_constant 32 : index
  %d3 = util.unfoldable_constant 16 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<?x?x32x16xi32>

  %packed_d0 = util.unfoldable_constant 100 : index
  %packed_d1 = util.unfoldable_constant 250 : index
  %init_unpack = tensor.empty(%packed_d0, %packed_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
  %cast_unpack = tensor.cast %unpack : tensor<?x?xi32> to tensor<100x250xi32>

  %source2 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %static_source = tensor.cast %source2 : tensor<?x?x?x?xi32> to tensor<4x16x32x16xi32>
  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [0, 2, 1, 3] -> [0, 1, 2, 3] is [0, 2, 1, 3].
  %transpose = linalg.transpose
    ins(%static_source: tensor<4x16x32x16xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [0, 2, 1, 3]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>
  %slice = tensor.extract_slice %collapse [0, 0] [100, 250] [1, 1] : tensor<128x256xi32> to tensor<100x250xi32>

  check.expect_eq(%cast_unpack, %slice) : tensor<100x250xi32>
  return
}

func.func @static_unpack_extract_slice_transpose_inner_dims_large() {
  %d0 = arith.constant 4 : index
  %d1 = arith.constant 16 : index
  %d2 = arith.constant 16 : index
  %d3 = arith.constant 32 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<4x16x16x32xi32>

  %init_unpack = tensor.empty() : tensor<100x250xi32>
  %unpack = linalg.unpack %source
      inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_unpack
      : tensor<4x16x16x32xi32> -> tensor<100x250xi32>

  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [0, 2, 3, 1] -> [0, 1, 2, 3] is [0, 3, 1, 2].
  %transpose = linalg.transpose
    ins(%source: tensor<4x16x16x32xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [0, 3, 1, 2]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>
  %slice = tensor.extract_slice %collapse [0, 0] [100, 250] [1, 1] : tensor<128x256xi32> to tensor<100x250xi32>

  check.expect_eq(%unpack, %slice) : tensor<100x250xi32>
  return
}

func.func @static_unpack_extract_slice_transpose_outer_dims_large() {
  %d0 = arith.constant 16 : index
  %d1 = arith.constant 4 : index
  %d2 = arith.constant 32 : index
  %d3 = arith.constant 16 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<16x4x32x16xi32>

  %init_unpack = tensor.empty() : tensor<100x250xi32>
  %unpack = linalg.unpack %source outer_dims_perm = [1, 0]  inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : tensor<16x4x32x16xi32> -> tensor<100x250xi32>

  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [2, 0, 1, 3] -> [0, 1, 2, 3] is [1, 2, 0, 3].
  %transpose = linalg.transpose
    ins(%source: tensor<16x4x32x16xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [1, 2, 0, 3]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>
  %slice = tensor.extract_slice %collapse [0, 0] [100, 250] [1, 1] : tensor<128x256xi32> to tensor<100x250xi32>

  check.expect_eq(%unpack, %slice) : tensor<100x250xi32>
  return
}

func.func @static_unpack_extract_slice_transpose_inner_and_outer_dims_large() {
  %d0 = arith.constant 16 : index
  %d1 = arith.constant 4 : index
  %d2 = arith.constant 16 : index
  %d3 = arith.constant 32 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<16x4x16x32xi32>

  %init_unpack = tensor.empty() : tensor<100x250xi32>
  %unpack = linalg.unpack %source
      outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_unpack
      : tensor<16x4x16x32xi32> -> tensor<100x250xi32>

  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [2, 0, 3, 1] -> [0, 1, 2, 3] is [1, 3, 0, 2].
  %transpose = linalg.transpose
    ins(%source: tensor<16x4x16x32xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [1, 3, 0, 2]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>
  %slice = tensor.extract_slice %collapse [0, 0] [100, 250] [1, 1] : tensor<128x256xi32> to tensor<100x250xi32>

  check.expect_eq(%unpack, %slice) : tensor<100x250xi32>

  return
}

func.func @dynamic_unpack_extract_slice_transpose_inner_dims_large() {
  %d0 = util.unfoldable_constant 4 : index
  %d1 = util.unfoldable_constant 16 : index
  %d2 = util.unfoldable_constant 16 : index
  %d3 = util.unfoldable_constant 32 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<?x?x16x32xi32>

  %packed_d0 = util.unfoldable_constant 100 : index
  %packed_d1 = util.unfoldable_constant 250 : index
  %init_unpack = tensor.empty(%packed_d0, %packed_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %source
      inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_unpack
      : tensor<?x?x16x32xi32> -> tensor<?x?xi32>
  %cast_unpack = tensor.cast %unpack : tensor<?x?xi32> to tensor<100x250xi32>

  %source2 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %static_source = tensor.cast %source2 : tensor<?x?x?x?xi32> to tensor<4x16x16x32xi32>
  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [0, 2, 3, 1] -> [0, 1, 2, 3] is [0, 3, 1, 2].
  %transpose = linalg.transpose
    ins(%static_source: tensor<4x16x16x32xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [0, 3, 1, 2]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>
  %slice = tensor.extract_slice %collapse [0, 0] [100, 250] [1, 1] : tensor<128x256xi32> to tensor<100x250xi32>

  check.expect_eq(%cast_unpack, %slice) : tensor<100x250xi32>
  return
}

func.func @dynamic_unpack_extract_slice_transpose_outer_dims_large() {
  %d0 = util.unfoldable_constant 16 : index
  %d1 = util.unfoldable_constant 4 : index
  %d2 = util.unfoldable_constant 32 : index
  %d3 = util.unfoldable_constant 16 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<?x?x32x16xi32>

  %packed_d0 = util.unfoldable_constant 100 : index
  %packed_d1 = util.unfoldable_constant 250 : index
  %init_unpack = tensor.empty(%packed_d0, %packed_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
  %cast_unpack = tensor.cast %unpack : tensor<?x?xi32> to tensor<100x250xi32>

  %source2 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %static_source = tensor.cast %source2 : tensor<?x?x?x?xi32> to tensor<16x4x32x16xi32>
  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [0, 2, 3, 1] -> [0, 1, 2, 3] is [0, 3, 1, 2].
  %transpose = linalg.transpose
    ins(%static_source: tensor<16x4x32x16xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [1, 2, 0, 3]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>
  %slice = tensor.extract_slice %collapse [0, 0] [100, 250] [1, 1] : tensor<128x256xi32> to tensor<100x250xi32>

  check.expect_eq(%cast_unpack, %slice) : tensor<100x250xi32>
  return
}

func.func @dynamic_unpack_extract_slice_transpose_inner_and_outer_dims_large() {
  %d0 = util.unfoldable_constant 16 : index
  %d1 = util.unfoldable_constant 4 : index
  %d2 = util.unfoldable_constant 16 : index
  %d3 = util.unfoldable_constant 32 : index
  %0 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %source = tensor.cast %0 : tensor<?x?x?x?xi32> to tensor<?x?x16x32xi32>

  %packed_d0 = util.unfoldable_constant 100 : index
  %packed_d1 = util.unfoldable_constant 250 : index
  %init_unpack = tensor.empty(%packed_d0, %packed_d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %source
      outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_unpack
      : tensor<?x?x16x32xi32> -> tensor<?x?xi32>
  %cast_unpack = tensor.cast %unpack : tensor<?x?xi32> to tensor<100x250xi32>

  %source2 = call @generate_4D_source(%d0, %d1, %d2, %d3) : (index, index, index, index) -> tensor<?x?x?x?xi32>
  %static_source = tensor.cast %source2 : tensor<?x?x?x?xi32> to tensor<16x4x16x32xi32>
  %init_transpose = tensor.empty() : tensor<4x32x16x16xi32>
  // The permutation for [2, 0, 3, 1] -> [0, 1, 2, 3] is [1, 3, 0, 2].
  %transpose = linalg.transpose
    ins(%static_source: tensor<16x4x16x32xi32>)
    outs(%init_transpose: tensor<4x32x16x16xi32>)
    permutation = [1, 3, 0, 2]
  %collapse = tensor.collapse_shape %transpose [[0, 1], [2, 3]] : tensor<4x32x16x16xi32> into tensor<128x256xi32>
  %slice = tensor.extract_slice %collapse [0, 0] [100, 250] [1, 1] : tensor<128x256xi32> to tensor<100x250xi32>

  check.expect_eq(%cast_unpack, %slice) : tensor<100x250xi32>
  return
}
