func @i32i32.i32() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<3> : tensor<2x4xi32>
  %rhs = iree.unfoldable_constant dense<2> : tensor<4x2xi32>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func @i8i8.i32() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<3> : tensor<2x4xi8>
  %rhs = iree.unfoldable_constant dense<2> : tensor<4x2xi8>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2x4xi8>, tensor<4x2xi8>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func @i16i16.i32() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<3> : tensor<2x4xi16>
  %rhs = iree.unfoldable_constant dense<2> : tensor<4x2xi16>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2x4xi16>, tensor<4x2xi16>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}
