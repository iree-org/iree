func.func @dynamic_dot() {
  %lhs = flow.tensor.dynamic_constant dense<[
    [15.0, 14.0, 13.0],
    [12.0, 11.0, 10.0],
    [09.0, 08.0, 07.0],
    [06.0, 05.0, 04.0],
    [03.0, 02.0, 01.0]]> : tensor<5x3xf32> -> tensor<?x?xf32>
  %rhs = flow.tensor.dynamic_constant dense<[
    [15.0, 14.0, 13.0, 12.0, 11.0],
    [10.0, 09.0, 08.0, 07.0, 06.0],
    [05.0, 04.0, 03.0, 02.0, 01.0]]> : tensor<3x5xf32> -> tensor<?x?xf32>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %dshape = util.optimization_barrier %res : tensor<?x?xf32>
  %result = tensor.cast %dshape : tensor<?x?xf32> to tensor<5x5xf32>
  check.expect_almost_eq_const(%result,
    dense<[[430.0, 388.0, 346.0, 304.0, 262.0],
           [340.0, 307.0, 274.0, 241.0, 208.0],
           [250.0, 226.0, 202.0, 178.0, 154.0],
           [160.0, 145.0, 130.0, 115.0, 100.0],
           [70.0, 64.0, 58.0, 52.0, 46.0]]> : tensor<5x5xf32>) : tensor<5x5xf32>
  return
}
