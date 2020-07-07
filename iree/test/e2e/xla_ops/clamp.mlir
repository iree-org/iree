func @i8() attributes { iree.module.export } {
  %min = iree.unfoldable_constant dense<[0, 0, 0, 0]> : tensor<4xi8>
  %val = iree.unfoldable_constant dense<[-2, 4, 8, 12]> : tensor<4xi8>
  %max = iree.unfoldable_constant dense<[10, 10, 10, 10]> : tensor<4xi8>
  %result = "mhlo.clamp"(%min, %val, %max) : (tensor<4xi8>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  check.expect_eq_const(%result, dense<[0, 4, 8, 10]> : tensor<4xi8>) : tensor<4xi8>
  return
}

func @i16() attributes { iree.module.export } {
  %min = iree.unfoldable_constant dense<[0, 0, 0, 0]> : tensor<4xi16>
  %val = iree.unfoldable_constant dense<[-2, 4, 8, 12]> : tensor<4xi16>
  %max = iree.unfoldable_constant dense<[10, 10, 10, 10]> : tensor<4xi16>
  %result = "mhlo.clamp"(%min, %val, %max) : (tensor<4xi16>, tensor<4xi16>, tensor<4xi16>) -> tensor<4xi16>
  check.expect_eq_const(%result, dense<[0, 4, 8, 10]> : tensor<4xi16>) : tensor<4xi16>
  return
}

func @i32() attributes { iree.module.export } {
  %min = iree.unfoldable_constant dense<[0, 0, 0, 0]> : tensor<4xi32>
  %val = iree.unfoldable_constant dense<[-2, 4, 8, 12]> : tensor<4xi32>
  %max = iree.unfoldable_constant dense<[10, 10, 10, 10]> : tensor<4xi32>
  %result = "mhlo.clamp"(%min, %val, %max) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[0, 4, 8, 10]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func @f32() attributes { iree.module.export } {
  %min = iree.unfoldable_constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>
  %val = iree.unfoldable_constant dense<[-2.0, 4.0, 8.0, 12.0]> : tensor<4xf32>
  %max = iree.unfoldable_constant dense<[10.0, 10.0, 10.0, 10.0]> : tensor<4xf32>
  %result = "mhlo.clamp"(%min, %val, %max) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_eq_const(%result, dense<[0.0, 4.0, 8.0, 10.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
