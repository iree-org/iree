// func @i32() attributes { iree.module.export } {
//   %lhs = iree.unfoldable_constant dense<3> : tensor<2x4xi32>
//   %rhs = iree.unfoldable_constant dense<2> : tensor<4x2xi32>
//   %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
//   check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
//   return
// }

func @i8() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<3> : tensor<2x4xi8>
  %rhs = iree.unfoldable_constant dense<2> : tensor<4x2xi8>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2x4xi8>, tensor<4x2xi8>) -> tensor<2x2xi8>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi8>) : tensor<2x2xi8>
  return
}
