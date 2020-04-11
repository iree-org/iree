func @tensor() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[1, 6, 7, 8]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 6, 3, 8]> : tensor<4xi32>
  %result = "xla_hlo.maximum"(%lhs, %rhs) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[5, 6, 7, 8]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func @tensor_odd_dim() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[1, 6, 7]> : tensor<3xi32>
  %rhs = iree.unfoldable_constant dense<[5, 6, 3]> : tensor<3xi32>
  %result = "xla_hlo.maximum"(%lhs, %rhs) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  check.expect_eq_const(%result, dense<[5, 6,7]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func @scalar() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i32>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i32>
  %result = "xla_hlo.maximum"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<2> : tensor<i32>) : tensor<i32>
  return
}

func @negative() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i32>
  %rhs = iree.unfoldable_constant dense<-2> : tensor<i32>
  %result = "xla_hlo.maximum"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<1> : tensor<i32>) : tensor<i32>
  return
}

func @i8() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i8>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i8>
  %result = "xla_hlo.maximum"(%lhs, %rhs) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%result, dense<2> : tensor<i8>) : tensor<i8>
  return
}

func @i16() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i16>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i16>
  %result = "xla_hlo.maximum"(%lhs, %rhs) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  check.expect_eq_const(%result, dense<2> : tensor<i16>) : tensor<i16>
  return
}

func @i64() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i64>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i64>
  %result = "xla_hlo.maximum"(%lhs, %rhs) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  check.expect_eq_const(%result, dense<2> : tensor<i64>) : tensor<i64>
  return
}
