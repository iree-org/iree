func @iota_dim0() attributes { iree.module.export } {
  %result = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<2x3xf32>
  check.expect_almost_eq_const(%result, dense<[
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}


func @iota_dim1() attributes { iree.module.export } {
  %result = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<2x3xf32>
  check.expect_almost_eq_const(%result, dense<[
    [0.0, 1.0, 2.0],
    [0.0, 1.0, 2.0]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}
