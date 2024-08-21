func.func @tensor() {
  %0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = util.unfoldable_constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %4 = stablehlo.compare EQ, %0, %1,  NOTYPE : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  stablehlo.custom_call @shape_assertion(%4) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<4xi1>) -> ()
  %result = "stablehlo.add"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[6.0, 8.0, 10.0, 12.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
