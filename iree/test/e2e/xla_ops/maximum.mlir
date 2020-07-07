func @tensor_i32() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[1, 6, 7, 8]> : tensor<4xi32>
  %rhs = iree.unfoldable_constant dense<[5, 6, 3, 8]> : tensor<4xi32>
  %result = "mhlo.maximum"(%lhs, %rhs) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[5, 6, 7, 8]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func @tensor_odd_dim() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[1, 6, 7]> : tensor<3xi32>
  %rhs = iree.unfoldable_constant dense<[5, 6, 3]> : tensor<3xi32>
  %result = "mhlo.maximum"(%lhs, %rhs) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  check.expect_eq_const(%result, dense<[5, 6,7]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func @scalar_i32() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i32>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i32>
  %result = "mhlo.maximum"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<2> : tensor<i32>) : tensor<i32>
  return
}

func @negative_i32() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i32>
  %rhs = iree.unfoldable_constant dense<-2> : tensor<i32>
  %result = "mhlo.maximum"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<1> : tensor<i32>) : tensor<i32>
  return
}

func @i8() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i8>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i8>
  %result = "mhlo.maximum"(%lhs, %rhs) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%result, dense<2> : tensor<i8>) : tensor<i8>
  return
}

func @i16() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i16>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i16>
  %result = "mhlo.maximum"(%lhs, %rhs) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  check.expect_eq_const(%result, dense<2> : tensor<i16>) : tensor<i16>
  return
}

func @i64() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1> : tensor<i64>
  %rhs = iree.unfoldable_constant dense<2> : tensor<i64>
  %result = "mhlo.maximum"(%lhs, %rhs) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  check.expect_eq_const(%result, dense<2> : tensor<i64>) : tensor<i64>
  return
}

func @tensor_f32() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[1.0, 2.0, 7.0, 4.0]> : tensor<4xf32>
  %rhs = iree.unfoldable_constant dense<[5.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "mhlo.minimum"(%lhs, %rhs) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func @scalar_f32() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %rhs = iree.unfoldable_constant dense<2.0> : tensor<f32>
  %result = "mhlo.minimum"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<1.0> : tensor<f32>) : tensor<f32>
  return
}

func @double() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f64>
  %rhs = iree.unfoldable_constant dense<2.0> : tensor<f64>
  %result = "mhlo.minimum"(%lhs, %rhs) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  check.expect_almost_eq_const(%result, dense<1.0> : tensor<f64>) : tensor<f64>
  return
}

func @negative_f32() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %rhs = iree.unfoldable_constant dense<-2.0> : tensor<f32>
  %result = "mhlo.minimum"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<-2.0> : tensor<f32>) : tensor<f32>
  return
}
