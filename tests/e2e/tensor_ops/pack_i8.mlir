func.func private @generate_2D_source(%height : index, %width : index) -> tensor<?x?xi8> {
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
      %c256 = arith.constant 256 : index
      %rem = arith.remui %linearized, %c256 : index
      %linearized_i8 = arith.index_cast %rem : index to i8
      linalg.yield %linearized_i8 : i8
  } -> tensor<?x?xi8>
  // This blocks the fusion for inputs and testing ops.
  %0 = util.optimization_barrier %source : tensor<?x?xi8>
  %1 = flow.tensor.tie_shape %0 : tensor<?x?xi8>{%height, %width}
  return %1 : tensor<?x?xi8>
}

func.func @static_pack_vnni_lhs_large() {
  %height = arith.constant 128 : index
  %width = arith.constant 256 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi8>
  %source = tensor.cast %0 : tensor<?x?xi8> to tensor<128x256xi8>

  %init_pack = tensor.empty() : tensor<8x128x16x2xi8>
  %pack = tensor.pack %source
    outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2]
    into %init_pack : tensor<128x256xi8> -> tensor<8x128x16x2xi8>

  // Pack without padding is just a reshape followed by a transpose.
  %reshape = tensor.expand_shape %source [[0, 1], [2, 3]] : tensor<128x256xi8> into tensor<8x16x128x2xi8>
  %init_transpose = tensor.empty() : tensor<8x128x16x2xi8>
  %transpose = linalg.transpose
    ins(%reshape : tensor<8x16x128x2xi8>)
    outs(%init_transpose : tensor<8x128x16x2xi8>)
    permutation = [0, 2, 1, 3]
  check.expect_eq(%pack, %transpose) : tensor<8x128x16x2xi8>
  return
}

func.func @static_pack_vnni_rhs_large() {
  %height = arith.constant 256 : index
  %width = arith.constant 512 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi8>
  %source = tensor.cast %0 : tensor<?x?xi8> to tensor<256x512xi8>

  %init_pack = tensor.empty() : tensor<32x128x16x2xi8>
  %pack = tensor.pack %source
    outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2]
    into %init_pack : tensor<256x512xi8> -> tensor<32x128x16x2xi8>

  // Pack without padding is just a reshape followed by a transpose.
  %reshape = tensor.expand_shape %source [[0, 1], [2, 3]] : tensor<256x512xi8> into tensor<128x2x32x16xi8>
  %init_transpose = tensor.empty() : tensor<32x128x16x2xi8>
  %transpose = linalg.transpose
    ins(%reshape : tensor<128x2x32x16xi8>)
    outs(%init_transpose : tensor<32x128x16x2xi8>)
    permutation = [2, 0, 3, 1]
  check.expect_eq(%pack, %transpose) : tensor<32x128x16x2xi8>
  return
}

func.func @static_pack_vnni_lhs_large_with_pad() {
  %height = arith.constant 127 : index
  %width = arith.constant 255 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi8>
  %source = tensor.cast %0 : tensor<?x?xi8> to tensor<127x255xi8>
  %c0_i8 = arith.constant 0 : i8

  %init_pack = tensor.empty() : tensor<8x128x16x2xi8>
  %pack = tensor.pack %source padding_value(%c0_i8 : i8)
    outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2]
    into %init_pack : tensor<127x255xi8> -> tensor<8x128x16x2xi8>

  %pad = tensor.pad %source low[0, 0] high[1, 1] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %c0_i8 : i8
  } : tensor<127x255xi8> to tensor<128x256xi8>
  %reshape = tensor.expand_shape %pad [[0, 1], [2, 3]] : tensor<128x256xi8> into tensor<8x16x128x2xi8>
  %init_transpose = tensor.empty() : tensor<8x128x16x2xi8>
  %transpose = linalg.transpose
    ins(%reshape : tensor<8x16x128x2xi8>)
    outs(%init_transpose : tensor<8x128x16x2xi8>)
    permutation = [0, 2, 1, 3]
  check.expect_eq(%pack, %transpose) : tensor<8x128x16x2xi8>
  return
}

func.func @static_pack_vnni_rhs_large_with_pad() {
  %height = arith.constant 255 : index
  %width = arith.constant 511 : index
  %0 = call @generate_2D_source(%height, %width) : (index, index) -> tensor<?x?xi8>
  %source = tensor.cast %0 : tensor<?x?xi8> to tensor<255x511xi8>
  %c0_i8 = arith.constant 0 : i8

  %init_pack = tensor.empty() : tensor<32x128x16x2xi8>
  %pack = tensor.pack %source padding_value(%c0_i8 : i8)
    outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2]
    into %init_pack : tensor<255x511xi8> -> tensor<32x128x16x2xi8>

  %pad = tensor.pad %source low[0, 0] high[1, 1] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %c0_i8 : i8
  } : tensor<255x511xi8> to tensor<256x512xi8>
  %reshape = tensor.expand_shape %pad [[0, 1], [2, 3]] : tensor<256x512xi8> into tensor<128x2x32x16xi8>
  %init_transpose = tensor.empty() : tensor<32x128x16x2xi8>
  %transpose = linalg.transpose
    ins(%reshape : tensor<128x2x32x16xi8>)
    outs(%init_transpose : tensor<32x128x16x2xi8>)
    permutation = [2, 0, 3, 1]
  check.expect_eq(%pack, %transpose) : tensor<32x128x16x2xi8>
  return
}
