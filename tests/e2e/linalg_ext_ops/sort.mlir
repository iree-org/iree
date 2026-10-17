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

// NaNs compare false in both directions with olt. Pin the legacy
// forward-only sweep, including the payload order for equal finite keys.
func.func @sort_nan_olt() {
  %keys = util.unfoldable_constant dense<[2.0, 0x7FC00000, 1.0, 1.0, 0.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %values : tensor<5xf32>, tensor<5xi32>) {
  ^bb0(%lhs: f32, %rhs: f32, %lhs_value: i32, %rhs_value: i32):
    %cmp = arith.cmpf olt, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<5xf32>, tensor<5xi32>
  // Compare bits so the NaN's position is checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[0, 2143289344, 1065353216, 1065353216, 1073741824]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[4, 1, 3, 2, 0]> : tensor<5xi32>) : tensor<5xi32>
  return
}

// NaNs compare false in both directions with ogt. Pin the legacy
// forward-only sweep, including the payload order for equal finite keys.
func.func @sort_nan_ogt() {
  %keys = util.unfoldable_constant dense<[0.0, 0x7FC00000, 1.0, 1.0, 2.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %values : tensor<5xf32>, tensor<5xi32>) {
  ^bb0(%lhs: f32, %rhs: f32, %lhs_value: i32, %rhs_value: i32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<5xf32>, tensor<5xi32>
  // Compare bits so the NaN's position is checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[1073741824, 2143289344, 1065353216, 1065353216, 0]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[4, 1, 3, 2, 0]> : tensor<5xi32>) : tensor<5xi32>
  return
}

// NaNs compare true in both directions with ult. Pin the legacy
// forward-only sweep, including the payload order for equal finite keys.
func.func @sort_nan_ult() {
  %keys = util.unfoldable_constant dense<[2.0, 0x7FC00000, 1.0, 1.0, 0.0]> : tensor<5xf32>
  %values = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %sort:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %values : tensor<5xf32>, tensor<5xi32>) {
  ^bb0(%lhs: f32, %rhs: f32, %lhs_value: i32, %rhs_value: i32):
    %cmp = arith.cmpf ult, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<5xf32>, tensor<5xi32>
  // Compare bits so the NaN's position is checked without floating-point equality.
  %bits = arith.bitcast %sort#0 : tensor<5xf32> to tensor<5xi32>
  check.expect_eq_const(%bits, dense<[1073741824, 2143289344, 0, 1065353216, 1065353216]> : tensor<5xi32>) : tensor<5xi32>
  check.expect_eq_const(%sort#1, dense<[0, 1, 4, 3, 2]> : tensor<5xi32>) : tensor<5xi32>
  return
}
