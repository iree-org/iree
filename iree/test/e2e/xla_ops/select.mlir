func @select() attributes { iree.module.export } {
  // TODO(b/132205704) support i1 in constants and function signatures.
  %input = iree.unfoldable_constant dense<[1, 0, 1, 0]> : tensor<4xi32>
  %zeros = iree.unfoldable_constant dense<0> : tensor<4xi32>
  %cond = "mhlo.compare"(%input, %zeros) {comparison_direction = "GT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %lhs = iree.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 6, 7, 8]> : tensor<4xi32>
  %result = "mhlo.select"(%cond, %lhs, %rhs) : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[1,6, 3, 8]> : tensor<4xi32>) : tensor<4xi32>
  return
}
