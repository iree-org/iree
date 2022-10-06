func.func @pack_simple() {
  %iree_input = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %init = linalg.init_tensor [2, 2, 2, 2] : tensor<2x2x2x2xi32>
  %pack = iree_linalg_ext.pack %iree_input dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : (tensor<4x4xi32> tensor<2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
  check.expect_eq_const(%pack, dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32>) : tensor<2x2x2x2xi32>
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
  // Pack without padding is just a reshape followed by a transpose.
  %reshape = tensor.expand_shape %source [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = linalg.init_tensor[4, 16, 32, 16] : tensor<4x16x32x16xi32>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%reshape : tensor<4x32x16x16xi32>) outs(%init_transpose : tensor<4x16x32x16xi32>) {
    ^bb0(%b0 : i32, %b1 : i32):
      linalg.yield %b0 : i32
    } -> tensor<4x16x32x16xi32>
  check.expect_eq(%pack, %transpose) : tensor<4x16x32x16xi32>
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
  %reshape = tensor.expand_shape %source [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = linalg.init_tensor[4, 16, 16, 32] : tensor<4x16x16x32xi32>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%reshape : tensor<4x32x16x16xi32>) outs(%init_transpose : tensor<4x16x16x32xi32>) {
    ^bb0(%b0 : i32, %b1 : i32):
      linalg.yield %b0 : i32
    } -> tensor<4x16x16x32xi32>
  check.expect_eq(%pack, %transpose) : tensor<4x16x16x32xi32>
  return
}

func.func @pack_pad_large() {
  %init_source = linalg.init_tensor [100, 250] : tensor<100x250xi32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init_source : tensor<100x250xi32>) {
    ^bb0(%b0 : i32):
      %outer = linalg.index 0 : index
      %inner = linalg.index 1 : index
      %c250 = arith.constant 250 : index
      %strided = arith.muli %outer, %c250 : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      linalg.yield %linearized_i32 : i32
  } -> tensor<100x250xi32>
  %padding_value = arith.constant 42 : i32
  %init_pack = linalg.init_tensor [4, 16, 32, 16] : tensor<4x16x32x16xi32>
  %pack = iree_linalg_ext.pack %source padding_value(%padding_value : i32)
      dims_pos = [0, 1] inner_tiles = [32, 16] into %init_pack
      : (tensor<100x250xi32> tensor<4x16x32x16xi32>) -> tensor<4x16x32x16xi32>
  %pad = tensor.pad %source low[0, 0] high[28, 6] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %padding_value : i32
  } : tensor<100x250xi32> to tensor<128x256xi32>
  %reshape = tensor.expand_shape %pad [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = linalg.init_tensor[4, 16, 32, 16] : tensor<4x16x32x16xi32>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%reshape : tensor<4x32x16x16xi32>) outs(%init_transpose : tensor<4x16x32x16xi32>) {
    ^bb0(%b0 : i32, %b1 : i32):
      linalg.yield %b0 : i32
    } -> tensor<4x16x32x16xi32>
  check.expect_eq(%pack, %transpose) : tensor<4x16x32x16xi32>
  return
}

func.func @pack_pad_transpose_large() {
  %init_source = linalg.init_tensor [100, 250] : tensor<100x250xi32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init_source : tensor<100x250xi32>) {
    ^bb0(%b0 : i32):
      %outer = linalg.index 0 : index
      %inner = linalg.index 1 : index
      %c250 = arith.constant 250 : index
      %strided = arith.muli %outer, %c250 : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      linalg.yield %linearized_i32 : i32
  } -> tensor<100x250xi32>
  %padding_value = arith.constant 42 : i32
  %init_pack = linalg.init_tensor [4, 16, 16, 32] : tensor<4x16x16x32xi32>
  %pack = iree_linalg_ext.pack %source padding_value(%padding_value : i32)
      dims_pos = [1, 0] inner_tiles = [16, 32] into %init_pack
      : (tensor<100x250xi32> tensor<4x16x16x32xi32>) -> tensor<4x16x16x32xi32>
  %pad = tensor.pad %source low[0, 0] high[28, 6] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %padding_value : i32
  } : tensor<100x250xi32> to tensor<128x256xi32>
  %reshape = tensor.expand_shape %pad [[0, 1], [2, 3]] : tensor<128x256xi32> into tensor<4x32x16x16xi32>
  %init_transpose = linalg.init_tensor[4, 16, 16, 32] : tensor<4x16x16x32xi32>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%reshape : tensor<4x32x16x16xi32>) outs(%init_transpose : tensor<4x16x16x32xi32>) {
    ^bb0(%b0 : i32, %b1 : i32):
      linalg.yield %b0 : i32
    } -> tensor<4x16x16x32xi32>
  check.expect_eq(%pack, %transpose) : tensor<4x16x16x32xi32>
  return
}
