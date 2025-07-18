func.func @softmax_static_10x256x256xf32() {
  %input = util.unfoldable_constant dense<5.000000e+00> : tensor<10x256x256xf32>
  %init = tensor.empty() : tensor<10x256x256xf32>
  %0 = linalg.softmax dimension(2) ins(%input : tensor<10x256x256xf32>) outs(%init : tensor<10x256x256xf32>) -> tensor<10x256x256xf32>
  check.expect_almost_eq_const(%0, dense<0.00390625> : tensor<10x256x256xf32>) : tensor<10x256x256xf32>
  return
}

func.func @softmax_dynamic_10x256x256xf32() {
  %input = flow.tensor.dynamic_constant dense<5.000000e+00> : tensor<10x256x256xf32> -> tensor<?x?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %input, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %input, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %input, %c2 : tensor<?x?x?xf32>
  %init = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %0 = linalg.softmax dimension(2) ins(%input : tensor<?x?x?xf32>) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %barrier = util.optimization_barrier %0 : tensor<?x?x?xf32>
  %result = tensor.cast %barrier : tensor<?x?x?xf32> to tensor<10x256x256xf32>
  check.expect_almost_eq_const(%result, dense<0.00390625> : tensor<10x256x256xf32>) : tensor<10x256x256xf32>
  return
}
