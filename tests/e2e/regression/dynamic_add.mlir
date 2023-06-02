func.func @main() {
  %lhs = flow.tensor.constant  dense<[[1.0,2.0,3.0,4.0],[-1.0,-2.0,-3.0,-4.0]]> : tensor<2x4xf32> -> tensor<?x4xf32>
  %rhs = flow.tensor.constant  dense<[[5.0,6.0,7.0,8.0],[-5.0,-6.0,-7.0,-8.0]]> : tensor<2x4xf32> -> tensor<?x4xf32>
  %2 = stablehlo.add %lhs, %rhs : tensor<?x4xf32>
  %3 = util.optimization_barrier %2 : tensor<?x4xf32>
  %result = tensor.cast %3 : tensor<?x4xf32> to tensor<2x4xf32>
  check.expect_almost_eq_const(%result, dense<[[6.0, 8.0, 10.0, 12.0],[-6.0, -8.0, -10.0, -12.0]]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}
