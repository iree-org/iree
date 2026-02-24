func.func @copy_like() {
  %source = util.unfoldable_constant dense<123.0> : tensor<4x16x64xf32>
  %output = tensor.empty() : tensor<4x16x64xf32>
  %padding = arith.constant 0.0 : f32
  %0 = iree_linalg_ext.map_load %source into %output {
    ^bb0(%idx0: index, %idx1: index, %idx2: index):
      iree_linalg_ext.yield %idx0, %idx1, %idx2, %padding : index, index, index, f32
  } : tensor<4x16x64xf32> into tensor<4x16x64xf32> -> tensor<4x16x64xf32>
  check.expect_almost_eq(%0, %source) : tensor<4x16x64xf32>
  return
}

func.func @expand_shape_like() {
  %source = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : tensor<16xf32>
  %padding = arith.constant 0.0 : f32
  %output = tensor.empty() : tensor<4x4xf32>
  %result = iree_linalg_ext.map_load %source into %output {
  ^bb0(%idx0: index, %idx1: index):
    %linear = affine.linearize_index disjoint [%idx0, %idx1] by (4, 4) : index
    iree_linalg_ext.yield %linear, %padding : index, f32
  } : tensor<16xf32> into tensor<4x4xf32> -> tensor<4x4xf32>
  %expected = tensor.expand_shape %source [[0, 1]] output_shape [4, 4] : tensor<16xf32> into tensor<4x4xf32>
  check.expect_almost_eq(%result, %expected) : tensor<4x4xf32>
  return
}

func.func @collapse_shape_like() {
  %source = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %padding = arith.constant 0 : i32
  %output = tensor.empty() : tensor<16xi32>
  %result = iree_linalg_ext.map_load %source into %output {
  ^bb0(%idx0: index):
    %2:2 = affine.delinearize_index %idx0 into (4, 4) : index, index
    iree_linalg_ext.yield %2#0, %2#1, %padding : index, index, i32
  } : tensor<4x4xi32> into tensor<16xi32> -> tensor<16xi32>
  %expected = tensor.collapse_shape %source [[0, 1]] : tensor<4x4xi32> into tensor<16xi32>
  check.expect_eq(%result, %expected) : tensor<16xi32>
  return
}

func.func @pad_slice_like() {
  // Source is 4 elements, output is 8 elements (with padding for out-of-bounds)
  %source = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %padding = arith.constant 0.0 : f32
  %output = tensor.empty() : tensor<8xf32>
  %result = iree_linalg_ext.map_load %source into %output {
  ^bb0(%idx0: index):
    // Identity mapping - indices 0-3 are in-bounds, 4-7 get padding
    iree_linalg_ext.yield %idx0, %padding : index, f32
  } : tensor<4xf32> into tensor<8xf32> -> tensor<8xf32>
  %expected = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]> : tensor<8xf32>
  check.expect_almost_eq(%result, %expected) : tensor<8xf32>
  return
}
