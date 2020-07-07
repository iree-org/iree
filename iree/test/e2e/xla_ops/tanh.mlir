func @tanh() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<
      [[-100.0, -5.0, -0.5,   1.0],
       [   1.2,  2.0,  3.0, 100.0]]> : tensor<2x4xf32>
  %result = "mhlo.tanh"(%input) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  check.expect_almost_eq_const(%result, dense<
      [[-1.0000, -0.9999, -0.4622, 0.7616],
       [ 0.8337,  0.9640,  0.9951, 1.0000]]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}
