// Test various forms of matmuls with narrow N, in particual matvec/batch_matvec
// (implicitly N=1) and matmuls with N=1 and N=2.
//
// The reason why this needs extensive e2e testing is the transposition of
// narrow N to narrow M in data tiling (around CPUMaterializeEncodingPass).
// It doesn't hurt to enable this case on all backends though.

func.func @matvec() {
  %lhs = util.unfoldable_constant dense<[
     [1, 2, 0, 5],
     [3, 4, -1, -3],
     [5, 6, -7, 0]
  ]> : tensor<3x4xi8>
  %rhs = util.unfoldable_constant dense<[-2, 3, 4, -1]> : tensor<4xi8>
  %acc = util.unfoldable_constant dense<[1, 2, 3]> : tensor<3xi32>
  %result = linalg.matvec ins(%lhs, %rhs : tensor<3x4xi8>, tensor<4xi8>) outs(%acc : tensor<3xi32>) -> tensor<3xi32>
  check.expect_eq_const(%result, dense<
    [0, 7, -17]
  > : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @batch_matvec() {
  %lhs = util.unfoldable_constant dense<[[
     [1, 2, 0, 5],
     [3, 4, -1, -3],
     [5, 6, -7, 0]
  ], [
     [-3, 1, 4, 2],
     [-1, 0, 6, -1],
     [1, -2, 3, -4]
  ]]> : tensor<2x3x4xi8>
  %rhs = util.unfoldable_constant dense<[
    [-2, 3, 4, -1],
    [1, 2, -5, 3]
  ]> : tensor<2x4xi8>
  %acc = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %result = linalg.batch_matvec ins(%lhs, %rhs : tensor<2x3x4xi8>, tensor<2x4xi8>) outs(%acc : tensor<2x3xi32>) -> tensor<2x3xi32>
  check.expect_eq_const(%result, dense<[
    [0, 7, -17],
    [-11, -29, -24]
  ]> : tensor<2x3xi32>) : tensor<2x3xi32>
  return
}

func.func @matmul_narrow_n_1() {
  %lhs = util.unfoldable_constant dense<[
     [1, 2, 0, 5],
     [3, 4, -1, -3],
     [5, 6, -7, 0]
  ]> : tensor<3x4xi8>
  %rhs = util.unfoldable_constant dense<[[-2], [3], [4], [-1]]> : tensor<4x1xi8>
  %acc = util.unfoldable_constant dense<[[1], [2], [3]]> : tensor<3x1xi32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<3x4xi8>, tensor<4x1xi8>) outs(%acc : tensor<3x1xi32>) -> tensor<3x1xi32>
  check.expect_eq_const(%result, dense<
    [[0], [7], [-17]]
  > : tensor<3x1xi32>) : tensor<3x1xi32>
  return
}

func.func @batch_matmul_narrow_n_1() {
  %lhs = util.unfoldable_constant dense<[[
     [1, 2, 0, 5],
     [3, 4, -1, -3],
     [5, 6, -7, 0]
  ], [
     [-3, 1, 4, 2],
     [-1, 0, 6, -1],
     [1, -2, 3, -4]
  ]]> : tensor<2x3x4xi8>
  %rhs = util.unfoldable_constant dense<[
    [[-2], [3], [4], [-1]],
    [[1], [2], [-5], [3]]
  ]> : tensor<2x4x1xi8>
  %acc = util.unfoldable_constant dense<[
    [[1], [2], [3]],
    [[4], [5], [6]]
  ]> : tensor<2x3x1xi32>
  %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x3x4xi8>, tensor<2x4x1xi8>) outs(%acc : tensor<2x3x1xi32>) -> tensor<2x3x1xi32>
  check.expect_eq_const(%result, dense<[
    [[0], [7], [-17]],
    [[-11], [-29], [-24]]
  ]> : tensor<2x3x1xi32>) : tensor<2x3x1xi32>
  return
}

func.func @matmul_narrow_n_2() {
  %lhs = util.unfoldable_constant dense<[
     [1, 2, 0, 5],
     [3, 4, -1, -3],
     [5, 6, -7, 0]
  ]> : tensor<3x4xi8>
  %rhs = util.unfoldable_constant dense<[[-2, 1], [3, -1], [4, 0], [-1, 2]]> : tensor<4x2xi8>
  %acc = util.unfoldable_constant dense<[[1, -1], [2, 0], [3, 1]]> : tensor<3x2xi32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<3x4xi8>, tensor<4x2xi8>) outs(%acc : tensor<3x2xi32>) -> tensor<3x2xi32>
  check.expect_eq_const(%result, dense<
    [[0, 8], [7, -7], [-17, 0]]
  > : tensor<3x2xi32>) : tensor<3x2xi32>
  return
}

func.func @batch_matmul_narrow_n_2() {
  %lhs = util.unfoldable_constant dense<[[
     [1, 2, 0, 5],
     [3, 4, -1, -3],
     [5, 6, -7, 0]
  ], [
     [-3, 1, 4, 2],
     [-1, 0, 6, -1],
     [1, -2, 3, -4]
  ]]> : tensor<2x3x4xi8>
  %rhs = util.unfoldable_constant dense<[
    [[-2, 0], [3, 1], [4, -1], [-1, 2]],
    [[1, -2], [2, 3], [-5, -3], [3, 0]]
  ]> : tensor<2x4x2xi8>
  %acc = util.unfoldable_constant dense<[
    [[1, -1], [2, 0], [3, 1]],
    [[4, 2], [5, 1], [6, -1]]
  ]> : tensor<2x3x2xi32>
  %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x3x4xi8>, tensor<2x4x2xi8>) outs(%acc : tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  check.expect_eq_const(%result, dense<[
    [[0, 11], [7, -1], [-17, 14]],
    [[-11, -1], [-29, -15], [-24, -18]]
  ]> : tensor<2x3x2xi32>) : tensor<2x3x2xi32>
  return
}
