// TODO(GH-1619): Remove the test and enable the test in xla_ops/
func @reduce_window_nonoverlapping_4x6xi32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[ 1,  2,  3,  4,  5,  6],
                                       [ 7,  8,  9, 10, 11, 12],
                                       [13, 14, 15, 16, 17, 18],
                                       [19, 20, 21, 22, 23, 24]]> : tensor<4x6xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>,
      window_strides = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x6xi32>, tensor<i32>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<[[30, 48],[102, 120]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}
