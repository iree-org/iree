func.func @copy_like() {
  %input = util.unfoldable_constant dense<123.0> : tensor<4x16x64xf32>
  %output = tensor.empty() : tensor<4x16x64xf32>
  %0 = iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index, %idx1: index, %idx2: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %idx2, %mask : index, index, index, i1
  } : tensor<4x16x64xf32> into tensor<4x16x64xf32> -> tensor<4x16x64xf32>
  check.expect_almost_eq(%0, %input) : tensor<4x16x64xf32>
  return
}

func.func @collapse_shape_like() {
  %input = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %true = arith.constant true
  %0 = tensor.empty() : tensor<16xi32>
  %result = iree_linalg_ext.map_scatter %input into %0 {
  ^bb0(%arg2: index, %arg3: index):
    %2 = affine.linearize_index disjoint [%arg2, %arg3] by (4, 4) : index
    iree_linalg_ext.yield %2, %true : index, i1
  } : tensor<4x4xi32> into tensor<16xi32> -> tensor<16xi32>
  %expected = tensor.collapse_shape %input [[0, 1]] : tensor<4x4xi32> into tensor<16xi32>
  check.expect_eq(%result, %expected) : tensor<16xi32>
  return
}

func.func @expand_shape_shape_like() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : tensor<16xf32>
  %true = arith.constant true
  %0 = tensor.empty() : tensor<4x4xf32>
  %result = iree_linalg_ext.map_scatter %input into %0 {
  ^bb0(%arg2: index):
    %2:2 = affine.delinearize_index %arg2 into (4, 4) : index, index
    iree_linalg_ext.yield %2#0, %2#1, %true : index, index, i1
  } : tensor<16xf32> into tensor<4x4xf32> -> tensor<4x4xf32>
  %expected = tensor.expand_shape %input [[0, 1]] output_shape [4, 4] : tensor<16xf32> into tensor<4x4xf32>
  check.expect_almost_eq(%result, %expected) : tensor<4x4xf32>
  return
}

func.func @extract_slice_like() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : tensor<16xf32>
  %c4 = arith.constant 4 : index
  %0 = tensor.empty() : tensor<4xf32>
  %result = iree_linalg_ext.map_scatter %input into %0 {
  ^bb0(%arg2: index):
    %2 = arith.cmpi ult, %arg2, %c4 : index
    iree_linalg_ext.yield %arg2, %2 : index, i1
  } : tensor<16xf32> into tensor<4xf32> -> tensor<4xf32>
  %expected = tensor.extract_slice %input[0] [4] [1] : tensor<16xf32> to tensor<4xf32>
  check.expect_almost_eq(%result, %expected) : tensor<4xf32>
  return
}
