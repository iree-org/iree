func @torch_select_index_0() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    [[01, 02, 03, 04, 05]],
    [[06, 07, 08, 09, 10]],
    [[11, 12, 13, 14, 15]],
    [[16, 17, 18, 19, 20]],
    [[21, 22, 23, 24, 25]]]> : tensor<5x1x5xi32>
  %indices = iree.unfoldable_constant dense<[0, 2]> : tensor<2xi32>
  %res = "mhlo.torch_index_select"(%input, %indices) {
    dim = 0 : i64,
    batch_dims = 0 : i64
  } : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  check.expect_eq_const(%res, dense<[[[01, 02, 03, 04, 05]], [[11, 12, 13, 14, 15]]]> : tensor<2x1x5xi32>) : tensor<2x1x5xi32>
  return
}

func @torch_select_index_1() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    [[ 1,  2],[ 3,  4]],
    [[ 5,  6],[ 7,  8]],
    [[ 9, 10],[11, 12]]]> : tensor<3x2x2xi32>
  %indices = iree.unfoldable_constant dense<[0, 1]> : tensor<2xi32>
  %res = "mhlo.torch_index_select"(%input, %indices) {
    dim = 1 : i64,
    batch_dims = 0 : i64
  } : (tensor<3x2x2xi32>, tensor<2xi32>) -> tensor<3x2x2xi32>
  check.expect_eq_const(%res, dense<[[[1,  2], [3,  4]], [[5,  6], [7,  8]],[[9, 10], [11, 12]]]> : tensor<3x2x2xi32>) : tensor<3x2x2xi32>
  return
}

func @torch_select_index_2() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    [[01, 02, 03, 04, 05]],
    [[06, 07, 08, 09, 10]],
    [[11, 12, 13, 14, 15]],
    [[16, 17, 18, 19, 20]],
    [[21, 22, 23, 24, 25]]]> : tensor<5x1x5xi32>
  %indices = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "mhlo.torch_index_select"(%input, %indices) {
    dim = 0 : i64,
    batch_dims = 0 : i64
  } : (tensor<5x1x5xi32>, tensor<i32>) -> tensor<1x5xi32>
  check.expect_eq_const(%res, dense<[[01, 02, 03, 04, 05]]> : tensor<1x5xi32>) : tensor<1x5xi32>
  return
}
