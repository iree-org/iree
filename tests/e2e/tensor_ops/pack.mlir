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

func.func @static_pack_simple() {
  %iree_input = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %init = tensor.empty() : tensor<2x2x2x2xi32>
  %pack = tensor.pack %iree_input inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : tensor<4x4xi32> -> tensor<2x2x2x2xi32>
  check.expect_eq_const(%pack, dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32>) : tensor<2x2x2x2xi32>
  return
}

func.func @dynamic_pack_simple() {
  %iree_input = flow.tensor.constant dense<[
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15]]> : tensor<4x4xi32> -> tensor<?x?xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %in_d0 = tensor.dim %iree_input, %c0 : tensor<?x?xi32>
  %in_d1 = tensor.dim %iree_input, %c1 : tensor<?x?xi32>
  %out_d0 = arith.ceildivui %in_d0, %c2 : index
  %out_d1 = arith.ceildivui %in_d1, %c2 : index
  %init = tensor.empty(%out_d0, %out_d1) : tensor<?x?x2x2xi32>
  %pack = tensor.pack %iree_input inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : tensor<?x?xi32> -> tensor<?x?x2x2xi32>
  %cast = tensor.cast %pack : tensor<?x?x2x2xi32> to tensor<2x2x2x2xi32>
  check.expect_eq_const(%cast, dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32>) : tensor<2x2x2x2xi32>
  return
}

func.func @static_pack_simple_pad_mode() {
  %iree_input = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %pad = arith.constant 0 : i32
  %init = tensor.empty() : tensor<2x2x3x3xi32>
  %pack = tensor.pack %iree_input padding_value(%pad : i32) inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %init
      : tensor<4x4xi32> -> tensor<2x2x3x3xi32>
  // After padding, the input is
  //  0,  1,  2,  3,  0,  0
  //  4,  5,  6,  7,  0,  0
  //  8,  9, 10, 11,  0,  0
  // 12, 13, 14, 15,  0,  0
  //  0,  0,  0,  0,  0,  0
  //  0,  0,  0,  0,  0,  0
  check.expect_eq_const(%pack, dense<[[[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                                       [[3, 0, 0], [7, 0, 0], [11, 0, 0]]],
                                      [[[12, 13, 14], [0, 0, 0], [0, 0, 0]],
                                       [[15, 0, 0], [0, 0, 0], [0, 0, 0]]]]> : tensor<2x2x3x3xi32>) : tensor<2x2x3x3xi32>
  return
}

func.func @dynamic_pack_simple_pad_mode() {
  %iree_input = flow.tensor.constant dense<[
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15]]> : tensor<4x4xi32> -> tensor<?x?xi32>
  %pad = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %in_d0 = tensor.dim %iree_input, %c0 : tensor<?x?xi32>
  %in_d1 = tensor.dim %iree_input, %c1 : tensor<?x?xi32>
  %out_d0 = arith.ceildivui %in_d0, %c3 : index
  %out_d1 = arith.ceildivui %in_d1, %c3 : index
  %init = tensor.empty(%out_d0, %out_d1) : tensor<?x?x3x3xi32>
  %pack = tensor.pack %iree_input padding_value(%pad : i32) inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %init
      : tensor<?x?xi32> -> tensor<?x?x3x3xi32>
  %cast = tensor.cast %pack : tensor<?x?x3x3xi32> to tensor<2x2x3x3xi32>
  check.expect_eq_const(%cast, dense<[[[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                                       [[3, 0, 0], [7, 0, 0], [11, 0, 0]]],
                                      [[[12, 13, 14], [0, 0, 0], [0, 0, 0]],
                                       [[15, 0, 0], [0, 0, 0], [0, 0, 0]]]]> : tensor<2x2x3x3xi32>) : tensor<2x2x3x3xi32>
  return
}

func.func @static_pack_large() {
  %height = arith.constant 128 : index
  %width = arith.constant 256 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<128x256xi32>

  %init_pack = tensor.empty() : tensor<4x16x32x16xi32>
  %pack = tensor.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_pack
      : tensor<128x256xi32> -> tensor<4x16x32x16xi32>

  // Pack without padding is just a reshape followed by a transpose.
  %reshape = tensor.expand_shape %source [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = tensor.empty() : tensor<4x16x32x16xi32>
  %transpose = linalg.transpose
    ins(%reshape : tensor<4x32x16x16xi32>)
    outs(%init_transpose : tensor<4x16x32x16xi32>)
    permutation = [0, 2, 1, 3]
  check.expect_eq(%pack, %transpose) : tensor<4x16x32x16xi32>
  return
}

func.func @static_pack_transpose_inner_dims_large() {
  %height = arith.constant 128 : index
  %width = arith.constant 256 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<128x256xi32>

  %init_pack = tensor.empty() : tensor<4x16x16x32xi32>
  %pack = tensor.pack %source inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_pack
      : tensor<128x256xi32> -> tensor<4x16x16x32xi32>
  %reshape = tensor.expand_shape %source [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = tensor.empty() : tensor<4x16x16x32xi32>
  %transpose = linalg.transpose
    ins(%reshape : tensor<4x32x16x16xi32>)
    outs(%init_transpose : tensor<4x16x16x32xi32>)
    permutation = [0, 2, 3, 1]

  check.expect_eq(%pack, %transpose) : tensor<4x16x16x32xi32>
  return
}

func.func @static_pack_pad_large() {
  %height = arith.constant 100 : index
  %width = arith.constant 250 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<100x250xi32>
  %padding_value = arith.constant 42 : i32

  %init_pack = tensor.empty() : tensor<4x16x32x16xi32>
  %pack = tensor.pack %source padding_value(%padding_value : i32)
      inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_pack
      : tensor<100x250xi32> -> tensor<4x16x32x16xi32>

  %pad = tensor.pad %source low[0, 0] high[28, 6] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %padding_value : i32
  } : tensor<100x250xi32> to tensor<128x256xi32>
  %reshape = tensor.expand_shape %pad [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = tensor.empty() : tensor<4x16x32x16xi32>
  %transpose = linalg.transpose
    ins(%reshape : tensor<4x32x16x16xi32>)
    outs(%init_transpose : tensor<4x16x32x16xi32>)
    permutation = [0, 2, 1, 3]

  check.expect_eq(%pack, %transpose) : tensor<4x16x32x16xi32>
  return
}

func.func @static_pack_pad_transpose_outer_dims_large() {
  %height = arith.constant 100 : index
  %width = arith.constant 250 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<100x250xi32>
  %padding_value = arith.constant 42 : i32

  %init_pack = tensor.empty() : tensor<16x4x32x16xi32>
  %pack = tensor.pack %source padding_value(%padding_value : i32)
      outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_pack
      : tensor<100x250xi32> -> tensor<16x4x32x16xi32>

  %pad = tensor.pad %source low[0, 0] high[28, 6] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %padding_value : i32
  } : tensor<100x250xi32> to tensor<128x256xi32>
  %reshape = tensor.expand_shape %pad [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = tensor.empty() : tensor<16x4x32x16xi32>
  %transpose = linalg.transpose
    ins(%reshape : tensor<4x32x16x16xi32>)
    outs(%init_transpose : tensor<16x4x32x16xi32>)
    permutation = [2, 0, 1, 3]

  check.expect_eq(%pack, %transpose) : tensor<16x4x32x16xi32>
  return
}

func.func @static_pack_pad_transpose_inner_dims_large() {
  %height = arith.constant 100 : index
  %width = arith.constant 250 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<100x250xi32>
  %padding_value = arith.constant 42 : i32

  %init_pack = tensor.empty() : tensor<4x16x16x32xi32>
  %pack = tensor.pack %source padding_value(%padding_value : i32)
      inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_pack
      : tensor<100x250xi32> -> tensor<4x16x16x32xi32>

  %pad = tensor.pad %source low[0, 0] high[28, 6] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %padding_value : i32
  } : tensor<100x250xi32> to tensor<128x256xi32>
  %reshape = tensor.expand_shape %pad [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = tensor.empty() : tensor<4x16x16x32xi32>
  %transpose = linalg.transpose
    ins(%reshape : tensor<4x32x16x16xi32>)
    outs(%init_transpose : tensor<4x16x16x32xi32>)
    permutation = [0, 2, 3, 1]

  check.expect_eq(%pack, %transpose) : tensor<4x16x16x32xi32>
  return
}

func.func @static_pack_pad_transpose_inner_and_outer_dims_large() {
  %height = arith.constant 100 : index
  %width = arith.constant 250 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<100x250xi32>
  %padding_value = arith.constant 42 : i32

  %init_pack = tensor.empty() : tensor<16x4x16x32xi32>
  %pack = tensor.pack %source padding_value(%padding_value : i32)
      outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_pack
      : tensor<100x250xi32> -> tensor<16x4x16x32xi32>

  %pad = tensor.pad %source low[0, 0] high[28, 6] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %padding_value : i32
  } : tensor<100x250xi32> to tensor<128x256xi32>
  %reshape = tensor.expand_shape %pad [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = tensor.empty() : tensor<16x4x16x32xi32>
  %transpose = linalg.transpose
    ins(%reshape : tensor<4x32x16x16xi32>)
    outs(%init_transpose : tensor<16x4x16x32xi32>)
    permutation = [2, 0, 3, 1]

  check.expect_eq(%pack, %transpose) : tensor<16x4x16x32xi32>
  return
}
