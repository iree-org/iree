func.func @dynamic_tensor() {
  %input = flow.tensor.constant dense<[[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]> : tensor<2x3xf32> -> tensor<?x?xf32>
  %res = stablehlo.abs %input : tensor<?x?xf32>
  %dshape = util.optimization_barrier %res : tensor<?x?xf32>
  %result = tensor.cast %dshape : tensor<?x?xf32> to tensor<2x3xf32>
  check.expect_almost_eq_const(%result, dense<[[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}
