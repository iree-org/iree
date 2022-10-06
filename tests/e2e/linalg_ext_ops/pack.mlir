func.func @pack_simple() {
  %iree_input = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %init = linalg.init_tensor [2, 2, 2, 2] : tensor<2x2x2x2xi32>
  %pack = iree_linalg_ext.pack %iree_input dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : (tensor<4x4xi32> tensor<2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
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
  %init = linalg.init_tensor [%out_d0, %out_d1, 2, 2] : tensor<?x?x2x2xi32>
  %pack = iree_linalg_ext.pack %iree_input dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : (tensor<?x?xi32> tensor<?x?x2x2xi32>) -> tensor<?x?x2x2xi32>
  %cast = tensor.cast %pack : tensor<?x?x2x2xi32> to tensor<2x2x2x2xi32>
  check.expect_eq_const(%cast, dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32>) : tensor<2x2x2x2xi32>
  return
}

func.func @pack_simple_pad_mode() {
  %iree_input = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %pad = arith.constant 0 : i32
  %init = linalg.init_tensor [2, 2, 3, 3] : tensor<2x2x3x3xi32>
  %pack = iree_linalg_ext.pack %iree_input padding_value(%pad : i32) dims_pos = [0, 1] inner_tiles = [3, 3] into %init
      : (tensor<4x4xi32> tensor<2x2x3x3xi32>) -> tensor<2x2x3x3xi32>
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
  %init = linalg.init_tensor [%out_d0, %out_d1, 3, 3] : tensor<?x?x3x3xi32>
  %pack = iree_linalg_ext.pack %iree_input padding_value(%pad : i32) dims_pos = [0, 1] inner_tiles = [3, 3] into %init
      : (tensor<?x?xi32> tensor<?x?x3x3xi32>) -> tensor<?x?x3x3xi32>
  %cast = tensor.cast %pack : tensor<?x?x3x3xi32> to tensor<2x2x3x3xi32>
  check.expect_eq_const(%cast, dense<[[[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                                       [[3, 0, 0], [7, 0, 0], [11, 0, 0]]],
                                      [[[12, 13, 14], [0, 0, 0], [0, 0, 0]],
                                       [[15, 0, 0], [0, 0, 0], [0, 0, 0]]]]> : tensor<2x2x3x3xi32>) : tensor<2x2x3x3xi32>
  return
}

func.func @pack_large() {
  %init_source = linalg.init_tensor [128, 256] : tensor<128x256xi32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init_source : tensor<128x256xi32>) {
    ^bb0(%b0 : i32):
      %outer = linalg.index 0 : index
      %inner = linalg.index 1 : index
      %c256 = arith.constant 256 : index
      %strided = arith.muli %outer, %c256 : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      linalg.yield %linearized_i32 : i32
  } -> tensor<128x256xi32>
  %init_pack = linalg.init_tensor [4, 16, 32, 16] : tensor<4x16x32x16xi32>
  %pack = iree_linalg_ext.pack %source dims_pos = [0, 1] inner_tiles = [32, 16] into %init_pack
      : (tensor<128x256xi32> tensor<4x16x32x16xi32>) -> tensor<4x16x32x16xi32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      outs(%init_pack : tensor<4x16x32x16xi32>) {
    ^bb0(%b0: i32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = linalg.index 2 : index
      %3 = linalg.index 3 : index
      %c32 = arith.constant 32 : index
      %outer_tile_strided = arith.muli %0, %c32 : index
      %outer = arith.addi %outer_tile_strided, %2 : index
      %c16 = arith.constant 16 : index
      %inner_tile_strided = arith.muli %1, %c16 : index
      %inner = arith.addi %inner_tile_strided, %3 : index
      %c256 = arith.constant 256 : index
      %strided = arith.muli %outer, %c256 : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      linalg.yield %linearized_i32 : i32
    } -> tensor<4x16x32x16xi32>
  check.expect_eq(%pack, %result) : tensor<4x16x32x16xi32>
  return
}

func.func @pack_transpose_large() {
  %init_source = linalg.init_tensor [128, 256] : tensor<128x256xi32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init_source : tensor<128x256xi32>) {
    ^bb0(%b0 : i32):
      %outer = linalg.index 0 : index
      %inner = linalg.index 1 : index
      %c256 = arith.constant 256 : index
      %strided = arith.muli %outer, %c256 : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      linalg.yield %linearized_i32 : i32
  } -> tensor<128x256xi32>
  %init_pack = linalg.init_tensor [4, 16, 16, 32] : tensor<4x16x16x32xi32>
  %pack = iree_linalg_ext.pack %source dims_pos = [1, 0] inner_tiles = [16, 32] into %init_pack
      : (tensor<128x256xi32> tensor<4x16x16x32xi32>) -> tensor<4x16x16x32xi32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      outs(%init_pack : tensor<4x16x16x32xi32>) {
    ^bb0(%b0: i32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = linalg.index 2 : index
      %3 = linalg.index 3 : index
      %c32 = arith.constant 32 : index
      %outer_tile_strided = arith.muli %0, %c32 : index
      %outer = arith.addi %outer_tile_strided, %3 : index
      %c16 = arith.constant 16 : index
      %inner_tile_strided = arith.muli %1, %c16 : index
      %inner = arith.addi %inner_tile_strided, %2 : index
      %c256 = arith.constant 256 : index
      %strided = arith.muli %outer, %c256 : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      linalg.yield %linearized_i32 : i32
    } -> tensor<4x16x16x32xi32>
  check.expect_eq(%pack, %result) : tensor<4x16x16x32xi32>
  return
}
