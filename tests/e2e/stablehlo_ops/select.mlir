func.func @select() {
  %input = util.unfoldable_constant dense<[1, 0, 1, 0]> : tensor<4xi1>
  %zeros = util.unfoldable_constant dense<0> : tensor<4xi1>
  %cond = "stablehlo.compare"(%input, %zeros) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  %lhs = util.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %rhs = util.unfoldable_constant dense<[5, 6, 7, 8]> : tensor<4xi32>
  %result = "stablehlo.select"(%cond, %lhs, %rhs) : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[1,6, 3, 8]> : tensor<4xi32>) : tensor<4xi32>
  return
}
