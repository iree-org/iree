func.func @sort2D() {
  %input = util.unfoldable_constant dense<[[5, 6], [3, 7]]> : tensor<2x2xi32>
  %empty = tensor.empty() : tensor<2x2xi32>
  %0 = iree_linalg_ext.sort dimension(0)
      ins(%input : tensor<2x2xi32>) outs(%empty : tensor<2x2xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):
      %1 = arith.cmpi slt, %arg2, %arg3 : i32
      iree_linalg_ext.yield %1 : i1
    } -> tensor<2x2xi32>
  check.expect_eq_const(%0, dense<[[3, 6], [5, 7]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

// Odd length with duplicate and negative keys, descending.
func.func @sort1D_duplicates() {
  %input = util.unfoldable_constant dense<[3.5, -1.0, 7.25, 0.0, 3.5, 12.0, -8.5, 2.0, 2.0, 9.0, -1.0, 5.5, 0.5, 7.25, -3.0, 1.0, 4.0]> : tensor<17xf32>
  %empty = tensor.empty() : tensor<17xf32>
  %0 = iree_linalg_ext.sort dimension(0)
      ins(%input : tensor<17xf32>) outs(%empty : tensor<17xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %1 = arith.cmpf ogt, %arg2, %arg3 : f32
      iree_linalg_ext.yield %1 : i1
    } -> tensor<17xf32>
  check.expect_eq_const(%0, dense<[12.0, 9.0, 7.25, 7.25, 5.5, 4.0, 3.5, 3.5, 2.0, 2.0, 1.0, 0.5, 0.0, -1.0, -1.0, -3.0, -8.5]> : tensor<17xf32>) : tensor<17xf32>
  return
}

func.func @sort1D_single_element() {
  %input = util.unfoldable_constant dense<[42]> : tensor<1xi32>
  %empty = tensor.empty() : tensor<1xi32>
  %0 = iree_linalg_ext.sort dimension(0)
      ins(%input : tensor<1xi32>) outs(%empty : tensor<1xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):
      %1 = arith.cmpi slt, %arg2, %arg3 : i32
      iree_linalg_ext.yield %1 : i1
    } -> tensor<1xi32>
  check.expect_eq_const(%0, dense<[42]> : tensor<1xi32>) : tensor<1xi32>
  return
}

// Sorts each row along the inner dimension.
func.func @sort2D_inner() {
  %input = util.unfoldable_constant dense<[[4, -2, 9, 0, 7], [1, 1, -5, 3, 2], [8, 6, 6, -1, 0]]> : tensor<3x5xi32>
  %empty = tensor.empty() : tensor<3x5xi32>
  %0 = iree_linalg_ext.sort dimension(1)
      ins(%input : tensor<3x5xi32>) outs(%empty : tensor<3x5xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):
      %1 = arith.cmpi slt, %arg2, %arg3 : i32
      iree_linalg_ext.yield %1 : i1
    } -> tensor<3x5xi32>
  check.expect_eq_const(%0, dense<[[-2, 0, 4, 7, 9], [-5, 1, 1, 2, 3], [-1, 0, 6, 6, 8]]> : tensor<3x5xi32>) : tensor<3x5xi32>
  return
}

// Argsort: the indices are moved together with the keys. The keys are
// distinct, so the result does not depend on how ties are ordered.
func.func @sort1D_keys_and_indices() {
  %keys = util.unfoldable_constant dense<[0.5, -2.0, 8.0, 3.0, -7.5, 6.0, 1.0, 4.5, 2.5]> : tensor<9xf32>
  %indices = util.unfoldable_constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8]> : tensor<9xi32>
  %empty_keys = tensor.empty() : tensor<9xf32>
  %empty_indices = tensor.empty() : tensor<9xi32>
  %0:2 = iree_linalg_ext.sort dimension(0)
      ins(%keys, %indices : tensor<9xf32>, tensor<9xi32>)
      outs(%empty_keys, %empty_indices : tensor<9xf32>, tensor<9xi32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i32, %arg5: i32):
      %1 = arith.cmpf ogt, %arg2, %arg3 : f32
      iree_linalg_ext.yield %1 : i1
    } -> tensor<9xf32>, tensor<9xi32>
  check.expect_eq_const(%0#0, dense<[8.0, 6.0, 4.5, 3.0, 2.5, 1.0, 0.5, -2.0, -7.5]> : tensor<9xf32>) : tensor<9xf32>
  check.expect_eq_const(%0#1, dense<[2, 5, 7, 3, 8, 6, 0, 1, 4]> : tensor<9xi32>) : tensor<9xi32>
  return
}
