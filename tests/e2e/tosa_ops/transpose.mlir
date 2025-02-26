func.func @test_transpose() {
  %0 = arith.constant dense<[[[0, 1, 2], [3, 4, 5]]]> : tensor<1x2x3xi32>
  %1 = tosa.transpose %0 { perms = array<i32: 1, 2, 0> }: (tensor<1x2x3xi32>) -> (tensor<2x3x1xi32>)
  check.expect_eq_const(%1, dense<[[[0], [1], [2]], [[3], [4], [5]]]> : tensor<2x3x1xi32>) : tensor<2x3x1xi32>
  return
}
