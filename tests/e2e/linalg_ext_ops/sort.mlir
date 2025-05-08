func.func @sort2D() {
  %input = util.unfoldable_constant dense<[[5, 6], [3, 7]]> : tensor<2x2xi32>
  %0 = iree_linalg_ext.sort dimension(0) outs(%input : tensor<2x2xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):
      %1 = arith.cmpi slt, %arg2, %arg3 : i32
      iree_linalg_ext.yield %1 : i1
    } -> tensor<2x2xi32>
  check.expect_eq_const(%0, dense<[[3, 6], [5, 7]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}
