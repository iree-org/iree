func.func @gather_from_splat_tensor() {
  %source = util.unfoldable_constant dense<0> : tensor<10x10xi32>
  %empty = tensor.empty() : tensor<1x10xi32>
  %indices = util.unfoldable_constant dense<0> : tensor<1xi32>
  %result = iree_linalg_ext.gather dimension_map = [0]
                          ins(%source, %indices : tensor<10x10xi32>, tensor<1xi32>)
                          outs(%empty : tensor<1x10xi32>) -> tensor<1x10xi32>

  check.expect_eq_const(%result, dense<0> : tensor<1x10xi32>)
            : tensor<1x10xi32>
  return
}

func.func @gather_2d_index_with_batch() {
  %source = util.unfoldable_constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %empty = tensor.empty() : tensor<2xi32>
  %indices = util.unfoldable_constant dense<[[0, 1], [1, 0]]> : tensor<2x2xi32>
  %result = iree_linalg_ext.gather dimension_map = [0, 1]
                          ins(%source, %indices : tensor<2x2xi32>, tensor<2x2xi32>)
                          outs(%empty: tensor<2xi32>) -> tensor<2xi32>
  check.expect_eq_const(%result, dense<[1, 2]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func @gather_2d_index_no_batch() {
  %source = util.unfoldable_constant dense<[[[0], [1]], [[0], [0]]]> : tensor<2x2x1xi32>
  %empty = tensor.empty() : tensor<1xi32>
  %indices = util.unfoldable_constant dense<[0, 1]> : tensor<2xi32>
  %result = iree_linalg_ext.gather dimension_map = [0, 1]
                          ins(%source, %indices : tensor<2x2x1xi32>, tensor<2xi32>)
                          outs(%empty: tensor<1xi32>) -> tensor<1xi32>
  check.expect_eq_const(%result, dense<[1]> : tensor<1xi32>) : tensor<1xi32>
  return
}

func.func @gather_1d_index_no_batch() {
  %source = util.unfoldable_constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %empty = tensor.empty() : tensor<2xi32>
  %indices = util.unfoldable_constant dense<[1]> : tensor<1xi32>
  %result = iree_linalg_ext.gather dimension_map = [0]
                          ins(%source, %indices : tensor<2x2xi32>, tensor<1xi32>)
                          outs(%empty: tensor<2xi32>) -> tensor<2xi32>
  check.expect_eq_const(%result, dense<[2, 3]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func @gather_perm_map() {
  %source = util.unfoldable_constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
  %empty = tensor.empty() : tensor<1xi32>
  %indices = util.unfoldable_constant dense<[0, 1]> : tensor<2xi32>
  %result = iree_linalg_ext.gather dimension_map = [1, 0]
                          ins(%source, %indices : tensor<2x2x1xi32>, tensor<2xi32>)
                          outs(%empty: tensor<1xi32>) -> tensor<1xi32>
  check.expect_eq_const(%result, dense<[2]> : tensor<1xi32>) : tensor<1xi32>
  return
}

func.func @gather_fuse_elementwise() {
  %cst = arith.constant 2 : i32
  %source = util.unfoldable_constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %empty = tensor.empty() : tensor<2xi32>
  %indices = util.unfoldable_constant dense<[1]> : tensor<1xi32>
  %result = iree_linalg_ext.gather dimension_map = [0]
                          ins(%source, %indices : tensor<2x2xi32>, tensor<1xi32>)
                          outs(%empty: tensor<2xi32>) -> tensor<2xi32>
  %generic = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%result : tensor<2xi32>) outs(%empty: tensor<2xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %0 = arith.muli %arg0, %cst : i32
      linalg.yield %0 : i32
  } -> tensor<2xi32>
  check.expect_eq_const(%generic, dense<[4, 6]> : tensor<2xi32>) : tensor<2xi32>
  return
}
