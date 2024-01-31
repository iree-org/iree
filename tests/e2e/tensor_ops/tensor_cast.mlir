func.func @tensor_cast() {
  %input = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %cast = tensor.cast %input : tensor<2x3xf32> to tensor<2x?xf32>
  %1 = util.optimization_barrier %cast : tensor<2x?xf32>
  %result = tensor.cast %cast : tensor<2x?xf32> to tensor<2x3xf32>
  check.expect_almost_eq_const(%result, dense<[[1.,2.,3.],[4.,5.,6.]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}
