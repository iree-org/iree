func.func @f32() {
  %0 = util.unfoldable_constant dense<[1.0, 6.0, -6.0, 0.0]> : tensor<4xf32>
  %1 = util.unfoldable_constant dense<[0.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %2 = "stablehlo.divide"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %result = "stablehlo.is_finite"(%2) : (tensor<4xf32>) -> tensor<4xi1>
  %c0 = util.unfoldable_constant dense<0> : tensor<4xi8>
  %c1 = util.unfoldable_constant dense<1> : tensor<4xi8>
  %output = "stablehlo.select"(%result, %c1, %c0) : (tensor<4xi1>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%output, dense<[0, 1, 1, 1]> : tensor<4xi8>) : tensor<4xi8>
  return
}
