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

func.func @unpack_simple() {
  %iree_input = util.unfoldable_constant dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32>
  %init = tensor.empty() : tensor<4x4xi32>
  %unpack = iree_linalg_ext.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : (tensor<2x2x2x2xi32> tensor<4x4xi32>) -> tensor<4x4xi32>
  check.expect_eq_const(%unpack, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}

func.func @dynamic_unpack_simple() {
  %iree_input = flow.tensor.constant dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32> -> tensor<?x?x2x2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %in_d0 = tensor.dim %iree_input, %c0 : tensor<?x?x2x2xi32>
  %in_d1 = tensor.dim %iree_input, %c1 : tensor<?x?x2x2xi32>
  %out_d0 = arith.muli %in_d0, %c2 : index
  %out_d1 = arith.muli %in_d1, %c2 : index
  %init = tensor.empty(%out_d0, %out_d1) : tensor<?x?xi32>
  %unpack = iree_linalg_ext.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : (tensor<?x?x2x2xi32> tensor<?x?xi32>) -> tensor<?x?xi32>
  %cast = tensor.cast %unpack : tensor<?x?xi32> to tensor<4x4xi32>
  check.expect_eq_const(%cast, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}

func.func @unpack_simple_extract_slice() {
  %iree_input = util.unfoldable_constant dense<[[[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                                       [[3, 0, 0], [7, 0, 0], [11, 0, 0]]],
                                      [[[12, 13, 14], [0, 0, 0], [0, 0, 0]],
                                       [[15, 0, 0], [0, 0, 0], [0, 0, 0]]]]> : tensor<2x2x3x3xi32>
  %init = tensor.empty() : tensor<4x4xi32>
  %unpack = iree_linalg_ext.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %init
      : (tensor<2x2x3x3xi32> tensor<4x4xi32>) -> tensor<4x4xi32>
  check.expect_eq_const(%unpack, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}

func.func @dynamic_unpack_simple_extract_slice() {
  %iree_input = flow.tensor.constant dense<[[[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
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
  %unpack = iree_linalg_ext.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %init
      : (tensor<?x?x3x3xi32> tensor<?x?xi32>) -> tensor<?x?xi32>
  %cast = tensor.cast %unpack : tensor<?x?xi32> to tensor<4x4xi32>
  check.expect_eq_const(%cast, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}

func.func @unpack_large() {
  %height = arith.constant 128 : index
  %width = arith.constant 256 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<128x256xi32>

  %init_pack = tensor.empty() : tensor<4x16x32x16xi32>
  %pack = iree_linalg_ext.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_pack
      : (tensor<128x256xi32> tensor<4x16x32x16xi32>) -> tensor<4x16x32x16xi32>

  %init_unpack = tensor.empty() : tensor<128x256xi32>
  %unpack = iree_linalg_ext.unpack %pack inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : (tensor<4x16x32x16xi32> tensor<128x256xi32>) -> tensor<128x256xi32>

  check.expect_eq(%unpack, %source) : tensor<128x256xi32>
  return
}

func.func @dynamic_unpack_large() {
  %d0 = util.unfoldable_constant 128 : index
  %d1 = util.unfoldable_constant 256 : index
  %source = call @generate_2D_source(%d0, %d1) : (index, index) -> tensor<?x?xi32>

  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %tiled_d0 = arith.ceildivui %d0, %c32 : index
  %tiled_d1 = arith.ceildivui %d1, %c16 : index
  %dyn_init_pack = tensor.empty(%tiled_d0, %tiled_d1) : tensor<?x?x32x16xi32>
  %pack = iree_linalg_ext.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dyn_init_pack
      : (tensor<?x?xi32> tensor<?x?x32x16xi32>) -> tensor<?x?x32x16xi32>

  %init_unpack = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %unpack = iree_linalg_ext.unpack %pack inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : (tensor<?x?x32x16xi32> tensor<?x?xi32>) -> tensor<?x?xi32>

  check.expect_eq(%unpack, %source) : tensor<?x?xi32>
  return
}

func.func @dynamic_unpack_transpose_large() {
  %d0 = util.unfoldable_constant 128 : index
  %d1 = util.unfoldable_constant 256 : index
  %source = call @generate_2D_source(%d0, %d1) : (index, index) -> tensor<?x?xi32>

  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %tiled_d0 = arith.ceildivui %d0, %c32 : index
  %tiled_d1 = arith.ceildivui %d1, %c16 : index
  %dyn_init_pack = tensor.empty(%tiled_d0, %tiled_d1) : tensor<?x?x16x32xi32>
  %pack = iree_linalg_ext.pack %source inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %dyn_init_pack
      : (tensor<?x?xi32> tensor<?x?x16x32xi32>) -> tensor<?x?x16x32xi32>

  %init_unpack = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %unpack = iree_linalg_ext.unpack %pack inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_unpack
      : (tensor<?x?x16x32xi32> tensor<?x?xi32>) -> tensor<?x?xi32>

  check.expect_eq(%unpack, %source) : tensor<?x?xi32>
  return
}

func.func @unpack_extract_slice_large() {
  %height = arith.constant 100 : index
  %width = arith.constant 250 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<100x250xi32>
  %padding_value = arith.constant 42 : i32

  %init_pack = tensor.empty() : tensor<4x16x32x16xi32>
  %pack = iree_linalg_ext.pack %source padding_value(%padding_value : i32)
      inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_pack
      : (tensor<100x250xi32> tensor<4x16x32x16xi32>) -> tensor<4x16x32x16xi32>

  %init_unpack = tensor.empty() : tensor<100x250xi32>
  %unpack = iree_linalg_ext.unpack %pack inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : (tensor<4x16x32x16xi32> tensor<100x250xi32>) -> tensor<100x250xi32>

  check.expect_eq(%unpack, %source) : tensor<100x250xi32>
  return
}

func.func @dynamic_unpack_extract_slice_large() {
  %d0 = util.unfoldable_constant 100 : index
  %d1 = util.unfoldable_constant 250 : index
  %source = call @generate_2D_source(%d0, %d1) : (index, index) -> tensor<?x?xi32>
  %padding_value = arith.constant 42 : i32

  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %tiled_d0 = arith.ceildivui %d0, %c32 : index
  %tiled_d1 = arith.ceildivui %d1, %c16 : index
  %dyn_init_pack = tensor.empty(%tiled_d0, %tiled_d1) : tensor<?x?x32x16xi32>
  %pack = iree_linalg_ext.pack %source padding_value(%padding_value : i32)
      inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dyn_init_pack
      : (tensor<?x?xi32> tensor<?x?x32x16xi32>) -> tensor<?x?x32x16xi32>

  %init_unpack = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %unpack = iree_linalg_ext.unpack %pack inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %init_unpack
      : (tensor<?x?x32x16xi32> tensor<?x?xi32>) -> tensor<?x?xi32>

  check.expect_eq(%unpack, %source) : tensor<?x?xi32>
  return
}

func.func @unpack_extract_slice_transpose_large() {
  %height = arith.constant 100 : index
  %width = arith.constant 250 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi32>
  %source = tensor.cast %0 : tensor<?x?xi32> to tensor<100x250xi32>
  %padding_value = arith.constant 42 : i32

  %init_pack = tensor.empty() : tensor<4x16x16x32xi32>
  %pack = iree_linalg_ext.pack %source padding_value(%padding_value : i32)
      inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_pack
      : (tensor<100x250xi32> tensor<4x16x16x32xi32>) -> tensor<4x16x16x32xi32>

  %init_unpack = tensor.empty() : tensor<100x250xi32>
  %unpack = iree_linalg_ext.unpack %pack
      inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_unpack
      : (tensor<4x16x16x32xi32> tensor<100x250xi32>) -> tensor<100x250xi32>

  check.expect_eq(%unpack, %source) : tensor<100x250xi32>
  return
}

func.func @dynamic_unpack_extract_slice_transpose_large() {
  %d0 = util.unfoldable_constant 100 : index
  %d1 = util.unfoldable_constant 250 : index
  %source = call @generate_2D_source(%d0, %d1) : (index, index) -> tensor<?x?xi32>
  %padding_value = arith.constant 42 : i32

  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %tiled_d0 = arith.ceildivui %d0, %c32 : index
  %tiled_d1 = arith.ceildivui %d1, %c16 : index
  %init_pack = tensor.empty(%tiled_d0, %tiled_d1) : tensor<?x?x16x32xi32>
  %pack = iree_linalg_ext.pack %source padding_value(%padding_value : i32)
      inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_pack
      : (tensor<?x?xi32> tensor<?x?x16x32xi32>) -> tensor<?x?x16x32xi32>

  %init_unpack = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %unpack = iree_linalg_ext.unpack %pack
      inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_unpack
      : (tensor<?x?x16x32xi32> tensor<?x?xi32>) -> tensor<?x?xi32>

  check.expect_eq(%unpack, %source) : tensor<?x?xi32>
  return
}
