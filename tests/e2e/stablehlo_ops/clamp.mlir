func.func @i8() {
  %min = util.unfoldable_constant dense<[0, 0, 0, 0]> : tensor<4xi8>
  %val = util.unfoldable_constant dense<[-2, 4, 8, 12]> : tensor<4xi8>
  %max = util.unfoldable_constant dense<[10, 10, 10, 10]> : tensor<4xi8>
  %result = "stablehlo.clamp"(%min, %val, %max) : (tensor<4xi8>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%result, dense<[0, 4, 8, 10]> : tensor<4xi8>) : tensor<4xi8>
  return
}

func.func @i16() {
  %min = util.unfoldable_constant dense<[0, 0, 0, 0]> : tensor<4xi16>
  %val = util.unfoldable_constant dense<[-2, 4, 8, 12]> : tensor<4xi16>
  %max = util.unfoldable_constant dense<[10, 10, 10, 10]> : tensor<4xi16>
  %result = "stablehlo.clamp"(%min, %val, %max) : (tensor<4xi16>, tensor<4xi16>, tensor<4xi16>) -> tensor<4xi16>
  check.expect_eq_const(%result, dense<[0, 4, 8, 10]> : tensor<4xi16>) : tensor<4xi16>
  return
}

func.func @i32() {
  %min = util.unfoldable_constant dense<[0, 0, 0, 0]> : tensor<4xi32>
  %val = util.unfoldable_constant dense<[-2, 4, 8, 12]> : tensor<4xi32>
  %max = util.unfoldable_constant dense<[10, 10, 10, 10]> : tensor<4xi32>
  %result = "stablehlo.clamp"(%min, %val, %max) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[0, 4, 8, 10]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func.func @f32() {
  %min = util.unfoldable_constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>
  %val = util.unfoldable_constant dense<[-2.0, 4.0, 8.0, 12.0]> : tensor<4xf32>
  %max = util.unfoldable_constant dense<[10.0, 10.0, 10.0, 10.0]> : tensor<4xf32>
  %result = "stablehlo.clamp"(%min, %val, %max) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_eq_const(%result, dense<[0.0, 4.0, 8.0, 10.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
