func.func @unpack_simple() {
  %iree_input = util.unfoldable_constant dense<[[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10 ,11], [14, 15]]]]> : tensor<2x2x2x2xi32>
  %init = tensor.empty() : tensor<4x4xi32>
  %unpack = iree_linalg_ext.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : (tensor<2x2x2x2xi32> tensor<4x4xi32>) -> tensor<4x4xi32>
  check.expect_eq_const(%unpack, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}

func.func @unpack_simple_extract_slice() {
  %iree_input = util.unfoldable_constant dense<[[[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                                       [[3, 0, 0], [7, 0, 0], [11, 0, 0]]],
                                      [[[12, 13, 14], [0, 0, 0], [0, 0, 0]],
                                       [[15, 0, 0], [0, 0, 0], [0, 0, 0]]]]> : tensor<2x2x3x3xi32>
  %pad = arith.constant 0 : i32
  %init = tensor.empty() : tensor<4x4xi32>
  %unpack = iree_linalg_ext.unpack %iree_input inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %init
      : (tensor<2x2x3x3xi32> tensor<4x4xi32>) -> tensor<4x4xi32>
  check.expect_eq_const(%unpack, dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>) : tensor<4x4xi32>
  return
}
